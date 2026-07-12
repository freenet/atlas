//! Minimal Freenet node client over the WebSocket command API, mirroring how
//! riverctl talks to a node with freenet-stdlib 0.8.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use freenet_stdlib::client_api::{
    ClientRequest, ContractRequest, ContractResponse, HostResponse, WebApi,
};
use freenet_stdlib::prelude::*;
use tokio_tungstenite::connect_async;

pub struct NodeClient {
    api: WebApi,
}

impl NodeClient {
    pub async fn connect(url: &str) -> Result<Self> {
        let (ws, _) = connect_async(url)
            .await
            .with_context(|| format!("connecting to {url}"))?;
        Ok(Self {
            api: WebApi::start(ws),
        })
    }

    async fn roundtrip(&mut self, req: ClientRequest<'static>) -> Result<HostResponse> {
        self.api.send(req).await.map_err(|e| anyhow!("send: {e}"))?;
        let resp = tokio::time::timeout(Duration::from_secs(60), self.api.recv())
            .await
            .map_err(|_| anyhow!("timed out waiting for node response"))?;
        resp.map_err(|e| anyhow!("node error: {e}"))
    }

    /// Compute the contract key for this contract code + parameters, no network.
    pub fn contract_key(code: &[u8], params: &[u8]) -> ContractKey {
        let contract_code = ContractCode::from(code.to_vec());
        ContractKey::from_params_and_code(Parameters::from(params.to_vec()), &contract_code)
    }

    pub async fn put(
        &mut self,
        code: &[u8],
        params: Vec<u8>,
        state: Vec<u8>,
        subscribe: bool,
    ) -> Result<ContractKey> {
        let contract_code = ContractCode::from(code.to_vec());
        let parameters = Parameters::from(params);
        let key = ContractKey::from_params_and_code(parameters.clone(), &contract_code);
        let container = ContractContainer::from(ContractWasmAPIVersion::V1(WrappedContract::new(
            Arc::new(contract_code),
            parameters,
        )));
        let req = ContractRequest::Put {
            contract: container,
            state: WrappedState::new(state),
            related_contracts: Default::default(),
            // Subscribe on publish so the node hosts the contract and it's
            // findable by cross-node GETs (a subscribe:false PUT stores it
            // locally but lets remote GETs dead-end).
            subscribe,
            blocking_subscribe: subscribe,
        };
        match self.roundtrip(ClientRequest::ContractOp(req)).await? {
            HostResponse::ContractResponse(ContractResponse::PutResponse { key }) => Ok(key),
            // PUT over an existing contract is applied as an update (e.g. a
            // web-container version bump), which the node acknowledges with an
            // update notification/response rather than a PutResponse.
            HostResponse::ContractResponse(ContractResponse::UpdateNotification { .. })
            | HostResponse::ContractResponse(ContractResponse::UpdateResponse { .. }) => Ok(key),
            HostResponse::Ok => Ok(key),
            other => Err(anyhow!("unexpected PUT response: {other:?}")),
        }
    }

    /// Raw GET of an arbitrary contract instance id (returns its state bytes).
    pub async fn get_instance(&mut self, id: ContractInstanceId) -> Result<Vec<u8>> {
        let req = ContractRequest::Get {
            key: id,
            return_contract_code: false,
            subscribe: false,
            blocking_subscribe: false,
        };
        match self.roundtrip(ClientRequest::ContractOp(req)).await? {
            HostResponse::ContractResponse(ContractResponse::GetResponse { state, .. }) => {
                Ok(state.as_ref().to_vec())
            }
            other => Err(anyhow!("unexpected GET response: {other:?}")),
        }
    }

    pub async fn get(&mut self, key: &ContractKey, subscribe: bool) -> Result<Vec<u8>> {
        let req = ContractRequest::Get {
            key: *key.id(),
            return_contract_code: false,
            subscribe,
            blocking_subscribe: subscribe,
        };
        match self.roundtrip(ClientRequest::ContractOp(req)).await? {
            HostResponse::ContractResponse(ContractResponse::GetResponse { state, .. }) => {
                Ok(state.as_ref().to_vec())
            }
            other => Err(anyhow!("unexpected GET response: {other:?}")),
        }
    }

    pub async fn update_delta(&mut self, key: ContractKey, delta: Vec<u8>) -> Result<()> {
        let req = ContractRequest::Update {
            key,
            data: UpdateData::Delta(StateDelta::from(delta)),
        };
        match self.roundtrip(ClientRequest::ContractOp(req)).await? {
            HostResponse::ContractResponse(ContractResponse::UpdateResponse { .. }) => Ok(()),
            HostResponse::Ok => Ok(()),
            other => Err(anyhow!("unexpected UPDATE response: {other:?}")),
        }
    }
}
