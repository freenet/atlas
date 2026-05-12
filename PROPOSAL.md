# RFC: Atlas - Decentralized Discovery Layer for Freenet

> **Status: Work in Progress — Early Draft**
>
> This document is an exploratory design sketch, not a final specification.
> Everything here — goals, contract architecture, descriptor schemas, ranking
> models, naming, scope — is open for discussion and likely to change.
> Nothing in this proposal should be treated as a committed design decision.
> Feedback, alternative proposals, and disagreement are welcome.

## Summary

Atlas is a proposed decentralized discovery layer for Freenet.

Its purpose is to allow users, applications, analyzers, and automated systems to describe, index, search, recommend, review, and discover arbitrary Freenet-addressable subjects without relying on a centralized search engine or recommendation service.

Atlas is not intended to provide a single canonical index or ranking system. Instead, it provides a framework for publishing signed metadata and building pluralistic discovery systems on top of that metadata.

The design is inspired partly by:

* early web directories (Yahoo, DMOZ),
* search engines (AltaVista, Google),
* recommendation systems,
* decentralized reputation systems,
* and systems that attach commentary to arbitrary web resources, such as Third Voice, Google SideWiki, Hypothes.is, Genius Web Annotator, and Dissenter/Gab.

Unlike those systems, Atlas is designed from the beginning as a decentralized, composable, contract-based system running natively on Freenet.

---

# User Experience

Open Atlas and it should feel quick, friendly, and immediately useful. Find what you're looking for. Discover things worth your time. No setup, no learning curve. Involvement is graduated: a tap on a recommendation tells you why it surfaced; a simple control lets you nudge what you see; someone who enjoys it can write or share a whole ranking recipe. None of that is required. The baseline is meant to be effortless and pleasant, with depth available for anyone who wants it.

Everything in the rest of this document (descriptors, analyzers, indexes, contract families, trust policies) is plumbing. An ordinary user should never need to know any of those terms exist.

---

# Goals

* Decentralized discovery of Freenet content and applications
* Human and machine-generated metadata
* Multiple competing indexes and ranking systems
* User-controlled trust and ranking policies
* Support for both manual curation and automated analysis
* Spam resistance without centralized moderation
* Extensible metadata model
* Ability to automate all interactions programmatically

---

# Non-Goals

Atlas is NOT:

* a single global search engine
* a single canonical recommendation algorithm
* a moderation authority
* a truth oracle
* a replacement for application-specific UIs
* a blockchain/token system

---

# Core Concepts

## Subjects

A subject is any entity that can be described or discovered.

Examples:

* Freenet web applications
* River rooms
* documents
* images
* videos
* audio
* contracts
* collections
* user profiles
* feeds
* external URLs
* software packages

The term "subject" is intentionally generic.

---

# Descriptors

A descriptor is a signed claim about a subject.

Examples:

* title
* summary
* tags
* screenshots
* embeddings
* reviews
* ratings
* corrections
* relationships
* safety notes
* curator recommendations

Descriptors may be:

* human-generated
* LLM-generated
* analyzer-generated
* curator-generated
* application-generated

Atlas does not assume descriptors are objectively true. Atlas only requires that they are attributable.

---

# Analyzers

Analyzers are systems which generate descriptors.

Examples:

* LLM summarizers
* screenshot generators
* transcript extractors
* embedding generators
* topic classifiers
* security analyzers

Analyzers should disclose:

* source code
* model/provider
* configuration
* prompt hash
* analyzer version

Example:

```text
analyzer_name
analyzer_version
source_code_url
model
prompt_hash
config
input_hashes
created_at
```

The intention is to make analyzer outputs inspectable and reproducible.

---

# Indexes

Indexes make subjects discoverable.

Atlas distinguishes analyzers from indexes.

Analyzers answer:

> "What is this?"

Indexes answer:

> "What matches this query?"

Examples:

* keyword indexes
* vector indexes
* curated collections
* recommendation indexes
* trending indexes

Indexes may draw from multiple analyzers.

Atlas does not require a single canonical index.

---

# Ranking

Atlas intentionally separates indexing from ranking.

Users may choose:

* trusted analyzers
* trusted indexes
* trusted curators
* ranking delegates
* local filtering policies

Possible ranking signals:

* analyzer trust
* GhostKey-backed feedback
* curator endorsements
* local interaction history
* pairwise preference data
* freshness
* diversity
* novelty

---

# GhostKeys Integration

GhostKeys are expected to play an important role in:

* spam resistance
* reputation persistence
* qualitative feedback
* trust weighting
* Sybil resistance

Atlas should integrate with the GhostKeys delegate so that:

* applications can request signatures
* users approve permissions
* credentials remain scoped
* private keys remain inaccessible to applications

Possible uses:

* ratings
* reviews
* correction claims
* trust assertions
* analyzer reputation

---

# Atlas UI

The default Atlas UI is intended to feel like an ordinary, modern app: browse, search, follow recommendations. Most users should never see the underlying machinery, and the system should work well for them without any configuration.

The full surface area is available on demand, for users who want it:

* see why a recommendation surfaced
* nudge what is shown
* follow or trust specific curators
* swap or share a ranking recipe
* submit, annotate, or review subjects
* see where any claim about a subject came from

The same capabilities are available to automated systems, so anything the UI can do can also be scripted.

---

# atlasctl

Atlas should also provide a command-line interface for automation.

Examples:

```bash
atlasctl submit <uri>
atlasctl describe <uri>
atlasctl review <subject>
atlasctl publish-descriptor descriptor.json
atlasctl search "decentralized chat"
atlasctl index build
```

This is expected to be the primary automation mechanism for:

* analyzers
* bots
* indexing systems
* research experiments
* recommendation systems
* external integrations

The Freenet-provided analyzer stack should itself use atlasctl.

---

# Contract Architecture

Atlas should NOT be implemented as a single contract.

Instead, Atlas should be a graph of many parameterized contract instances.

This is similar to how River models rooms as parameterized contracts.

Contract identity is determined by:

* contract Wasm hash
* constructor-like parameters

Different parameters imply different contract identities.

This allows Atlas to scale horizontally and provide multiple access patterns.

---

# Proposed Contract Families

## Subject Contracts

Parameters:

```text
subject_hash
```

Purpose:

* descriptors about one subject
* reviews
* ratings
* corrections
* relationships

---

## Analyzer Feed Contracts

Parameters:

```text
analyzer_public_key
```

Purpose:

* descriptors published by one analyzer
* provenance stream
* analyzer history

---

## Keyword Index Shard Contracts

Parameters:

```text
indexer_key
shard_id
```

Purpose:

* token -> subject references
* keyword search

---

## Vector Index Contracts

Parameters:

```text
indexer_key
partition_id
embedding_model
```

Purpose:

* semantic/vector search

---

## Collection Contracts

Parameters:

```text
curator_key
collection_slug
```

Purpose:

* curated lists
* recommendations
* human discovery

---

## Feedback Contracts

Parameters:

```text
user_or_ghostkey_identity
```

Purpose:

* ratings
* reviews
* trust assertions
* pairwise preference data

---

# Storage Philosophy

Atlas should prefer:

* append-only records
* signed claims
* mergeable state
* bounded/prunable indexes

Atlas should avoid:

* mutable canonical truth
* centralized moderation logic
* globally authoritative metadata

---

# Descriptor Types

Potential descriptor categories:

## Descriptive

* title
* summary
* tags
* screenshots
* embeddings

## Relational

* similar_to
* belongs_to_collection
* replies_to
* references

## Evaluative

* rating
* recommendation
* quality assessment
* spam assessment

## Corrective

* inaccurate_summary
* stale_screenshot
* disputed_claim

---

# Feedback Models

Atlas should not prematurely standardize on one rating mechanism.

Potential models:

* 1-5 stars
* thumbs up/down
* pairwise preference
* useful/not useful
* spam/not spam
* freeform review

Pairwise comparisons may be especially useful for training recommendation/ranking systems.

---

# Design Principles

## 1. Separate Description From Ranking

Atlas should not decide what deserves attention.

Atlas should allow many ranking systems to coexist.

---

## 2. Preserve Provenance

Descriptors should remain attributable.

Indexes aggregate claims rather than replacing them.

---

## 3. Human + Machine Collaboration

Atlas should support:

* human curation
* automated analyzers
* LLM-generated metadata
* social discovery
* local ranking

---

## 4. Legibility Over Suppression

Atlas should focus on helping users understand what things are.

Accurate description may reduce the need for aggressive centralized moderation.

---

# Open Questions

* Best descriptor schema?
* Best contract partitioning strategy?
* How should vector indexes be sharded?
* How should trust policies be represented?
* How should GhostKey weighting work?
* How should pruning/bounded state work?
* Should analyzers publish raw embeddings or references?
* How should local/private ranking signals interact with public metadata?
* Should Atlas support anonymous descriptors?
* How should recommendation algorithms be shared/distributed?

---

# Initial Implementation Proposal

V1 should intentionally remain modest.

Suggested initial scope:

* subject submission
* analyzer-generated summaries/tags
* screenshots
* keyword search
* curated collections
* simple local ranking
* atlasctl
* GhostKey-backed reviews optional

Avoid trying to solve:

* global reputation
* fully decentralized semantic search
* universal recommendation
* complex trust propagation
* automated moderation

The initial goal should be:

> Make Freenet content discoverable and navigable.
