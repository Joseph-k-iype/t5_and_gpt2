# SPEC-04 — Search Pipeline, Embeddings, and AI Synthesis
**EA Knowledge Hub v2 · Claude Code implementation spec**

---

## 1. Pipeline architecture

```
User query
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Query Understanding Layer                      │
│  Intent classify → Entity extract → Coord infer │
└─────────────────────┬───────────────────────────┘
                      │
        ┌─────────────┼───────────────┐
        ▼             ▼               ▼
  ┌──────────┐  ┌──────────┐  ┌──────────────────┐
  │  BM25    │  │ Semantic │  │ Graph traversal  │
  │  (ES)    │  │  KNN(ES) │  │  (FalkorDB)      │
  └────┬─────┘  └────┬─────┘  └────────┬─────────┘
       └─────────────┼─────────────────┘
                     ▼
          ┌─────────────────────┐
          │  Hybrid Fusion(RRF) │
          │  + Re-ranking       │
          └──────────┬──────────┘
                     │
         ┌───────────┴──────────────┐
         ▼                          ▼
  ┌────────────────┐     ┌─────────────────────┐
  │ Ranked results │     │   AI Synthesis      │
  │ + dynamic facets│    │   (LLM RAG, top-5)  │
  └────────────────┘     └─────────────────────┘
```

---

## 2. Query understanding

```python
# services/search_service.py — QueryUnderstanding class

class QueryUnderstanding:

    INTENT_PATTERNS = {
        'lookup':   r'(find|show|get|what is|define|explain|ADR-\d+)',
        'evaluate': r'(compare|vs|versus|better|recommend|choose|select|which)',
        'browse':   r'(list|what.*patterns?|show.*all|browse)',
        'learn':    r'(how does|overview|understand)',
    }

    TOGAF_DOMAIN_HINTS = {
        'capability': 'Business', 'process': 'Business', 'value chain': 'Business',
        'data': 'Data', 'lineage': 'Data', 'master data': 'Data', 'lakehouse': 'Data',
        'integration': 'Application', 'microservice': 'Application', 'api': 'Application',
        'event': 'Application', 'saga': 'Application', 'cqrs': 'Application',
        'cloud': 'Technology', 'kubernetes': 'Technology', 'network': 'Technology',
        'platform': 'Technology', 'infrastructure': 'Technology',
        'ai': 'Intelligence', 'ml': 'Intelligence', 'rag': 'Intelligence',
        'model': 'Intelligence', 'embedding': 'Intelligence',
    }

    ADM_PHASE_HINTS = {
        'vision': 'A', 'strategy': 'A',
        'business architecture': 'B', 'capability map': 'B',
        'information systems': 'C', 'application architecture': 'C',
        'technology architecture': 'D',
        'migration': 'E', 'transition': 'E',
        'implementation governance': 'G', 'compliance': 'G',
        'change management': 'H', 'lifecycle': 'H',
    }

    ZACHMAN_ROW_HINTS = {
        'executive': 'Planner', 'cto': 'Planner', 'strategy': 'Planner',
        'business': 'Owner', 'domain owner': 'Owner',
        'architect': 'Designer', 'solution': 'Designer',
        'engineer': 'Builder', 'platform': 'Builder',
        'dba': 'Subcontractor', 'ops': 'Subcontractor',
    }

    ZACHMAN_COL_HINTS = {
        'data': 'What', 'entity': 'What', 'schema': 'What',
        'process': 'How', 'function': 'How', 'integration': 'How',
        'deploy': 'Where', 'region': 'Where', 'jurisdiction': 'Where',
        'team': 'Who', 'access': 'Who', 'identity': 'Who',
        'event': 'When', 'sequence': 'When', 'timeline': 'When',
        'rule': 'Why', 'compliance': 'Why', 'regulation': 'Why',
    }

    REGULATORY_HINTS = {
        'gdpr': 'GDPR', 'schrems': 'GDPR',
        'mas': 'MAS_TRM', 'mas trm': 'MAS_TRM',
        'hkma': 'HKMA',
        'dpdpa': 'DPDPA',
        'basel': 'Basel_III', 'bcbs': 'BCBS239',
        'psd2': 'PSD2',
        'fca': 'FCA',
    }

    async def understand(self, query: str, user: User) -> QueryContext:
        q = query.lower()
        intent = self._classify_intent(q)
        togaf_domains = self._extract_hints(q, self.TOGAF_DOMAIN_HINTS)
        adm_phases = self._extract_hints(q, self.ADM_PHASE_HINTS)
        zachman_rows = self._extract_hints(q, self.ZACHMAN_ROW_HINTS)
        zachman_cols = self._extract_hints(q, self.ZACHMAN_COL_HINTS)
        regulatory_tags = self._extract_hints(q, self.REGULATORY_HINTS)

        # Fallback: use user's persona preference as Zachman row hint
        if not zachman_rows and user.preferences.get('persona_view'):
            zachman_rows = [user.preferences['persona_view']]

        return QueryContext(
            original=query,
            intent=intent,
            togaf_domains=list(set(togaf_domains)),
            adm_phases=list(set(adm_phases)),
            zachman_rows=list(set(zachman_rows)),
            zachman_cols=list(set(zachman_cols)),
            regulatory_tags=list(set(regulatory_tags)),
            jurisdiction_context=user.preferences.get('jurisdiction_context'),
        )

    def _extract_hints(self, query: str, hint_map: dict) -> list:
        return list({v for k, v in hint_map.items() if k in query})

    def _classify_intent(self, query: str) -> str:
        import re
        for intent, pattern in self.INTENT_PATTERNS.items():
            if re.search(pattern, query):
                return intent
        return 'browse'
```

---

## 3. BM25 retrieval (Elasticsearch)

```python
async def bm25_search(self, query_ctx: QueryContext,
                       filters: SearchFilters, size: int = 50) -> list[SearchHit]:
    must_filters = self._build_filter_clauses(filters, query_ctx)

    body = {
        "size": size,
        "query": {
            "bool": {
                "must": [{
                    "multi_match": {
                        "query": query_ctx.original,
                        "fields": [
                            "name^3",
                            "summary^2",
                            "problem_statement^2",
                            "decision_statement^2",
                            "content_body",
                            "consequences_gains",
                            "consequences_tradeoffs"
                        ],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                }],
                "filter": must_filters,
                "should": [
                    {"term": {"certification_tier": {"value": "Mandated", "boost": 1.3}}},
                    {"term": {"certification_tier": {"value": "Certified", "boost": 1.0}}},
                    {"term": {"certification_tier": {"value": "Reviewed", "boost": 0.7}}},
                    *[{"term": {"togaf_domains": {"value": d, "boost": 1.2}}}
                      for d in (query_ctx.togaf_domains or [])],
                    *[{"term": {"zachman_rows": {"value": r, "boost": 1.1}}}
                      for r in (query_ctx.zachman_rows or [])],
                ]
            }
        },
        "highlight": {
            "fields": {
                "summary": {},
                "problem_statement": {},
                "content_body": {"number_of_fragments": 1, "fragment_size": 200}
            }
        }
    }
    response = await es.search(index="ea_hub_artifacts", body=body)
    return self._parse_es_hits(response, source='bm25')

def _build_filter_clauses(self, filters: SearchFilters, ctx: QueryContext) -> list:
    clauses = [
        {"terms": {"status": ["Reviewed", "Accepted", "Certified", "Under_Watch", "Mandated"]}}
    ]
    if not filters.include_archived:
        clauses.append({"terms": {"status": ["Reviewed","Accepted","Certified","Mandated","Under_Watch"]}})
    if filters.content_type:
        clauses.append({"terms": {"content_type": filters.content_type}})
    if filters.certification:
        clauses.append({"terms": {"certification_tier": filters.certification}})
    if filters.togaf_domain:
        clauses.append({"terms": {"togaf_domains": filters.togaf_domain}})
    if filters.adm_phases:
        clauses.append({"terms": {"adm_phases": filters.adm_phases}})
    if filters.zachman_rows:
        clauses.append({"terms": {"zachman_rows": filters.zachman_rows}})
    if filters.zachman_cols:
        clauses.append({"terms": {"zachman_cols": filters.zachman_cols}})
    if filters.jurisdictions:
        clauses.append({"terms": {"applicable_jurisdictions": filters.jurisdictions}})
    if filters.regulatory_tags:
        clauses.append({"terms": {"regulatory_tags": filters.regulatory_tags}})
    if filters.technology_tags:
        clauses.append({"terms": {"technology_tags": filters.technology_tags}})
    if filters.min_fitness:
        clauses.append({"range": {"fitness_score": {"gte": filters.min_fitness}}})
    return clauses
```

---

## 4. Semantic KNN retrieval (Elasticsearch)

```python
async def semantic_search(self, query_ctx: QueryContext,
                           filters: SearchFilters, size: int = 50) -> list[SearchHit]:
    query_embedding = await embedding_service.embed(query_ctx.original)
    filter_clauses = self._build_filter_clauses(filters, query_ctx)

    body = {
        "size": size,
        "knn": {
            "field": "embedding_vector",
            "query_vector": query_embedding,
            "k": size,
            "num_candidates": size * 10,
            "filter": {"bool": {"must": filter_clauses}}
        }
    }
    response = await es.search(index="ea_hub_artifacts", body=body)
    return self._parse_es_hits(response, source='semantic')
```

---

## 5. Graph traversal retrieval (FalkorDB)

```python
async def graph_search(self, query_ctx: QueryContext,
                        filters: SearchFilters,
                        bm25_ids: list[str], size: int = 30) -> list[SearchHit]:
    """Start from BM25 top hits, traverse to surface related artifacts."""
    if not bm25_ids:
        return []
    seed_ids = bm25_ids[:10]

    cypher = """
    UNWIND $seed_ids AS seed_id
    MATCH (seed:Artifact {id: seed_id})-[r*1..2]-(related:Artifact)
    WHERE related.id NOT IN $seed_ids
      AND related.status IN ['Certified', 'Mandated', 'Reviewed']
      AND related.fitness_score >= $min_fitness
    RETURN DISTINCT related, type(r[0]) AS rel_type, min(length(r)) AS hops
    ORDER BY hops ASC, related.fitness_score DESC
    LIMIT $size
    """
    result = await falkordb.query(cypher, {
        'seed_ids': seed_ids,
        'min_fitness': filters.min_fitness or 0.0,
        'size': size
    })
    hits = []
    for row in result.result_set:
        artifact_node = row[0]
        hops = row[2]
        hits.append(SearchHit(
            artifact_id=artifact_node['id'],
            score=0.5 / hops,  # 1-hop=0.5, 2-hop=0.25
            source='graph',
            rel_type=row[1],
            match_snippet=f"Related via {row[1].replace('_', ' ').lower()}"
        ))
    return hits
```

---

## 6. Hybrid fusion and re-ranking

```python
def reciprocal_rank_fusion(self,
                            bm25: list[SearchHit],
                            semantic: list[SearchHit],
                            graph: list[SearchHit],
                            k: int = 60) -> list[SearchHit]:
    scores: dict[str, float] = {}
    all_hits: dict[str, SearchHit] = {}

    for rank, hit in enumerate(bm25):
        scores[hit.artifact_id] = scores.get(hit.artifact_id, 0) + 1 / (k + rank + 1)
        all_hits[hit.artifact_id] = hit

    for rank, hit in enumerate(semantic):
        scores[hit.artifact_id] = scores.get(hit.artifact_id, 0) + 1 / (k + rank + 1)
        all_hits.setdefault(hit.artifact_id, hit)

    for rank, hit in enumerate(graph):
        # Graph hits weighted 0.5× (discovery signal)
        scores[hit.artifact_id] = scores.get(hit.artifact_id, 0) + 0.5 / (k + rank + 1)
        all_hits.setdefault(hit.artifact_id, hit)

    return sorted(
        [SearchHit(**{**vars(all_hits[aid]), 'score': score})
         for aid, score in scores.items()],
        key=lambda h: h.score, reverse=True
    )

def apply_reranking_multipliers(self,
                                 hits: list[SearchHit],
                                 query_ctx: QueryContext,
                                 user: User,
                                 artifacts: dict[str, Artifact]) -> list[SearchHit]:
    for hit in hits:
        a = artifacts.get(hit.artifact_id)
        if not a:
            hit.final_score = hit.score * 0.1
            continue
        m = 1.0

        # Fitness multiplier
        m *= max(0.5, min(1.2, a.fitness_score / 80))

        # Certification tier
        m *= {'Community': 0.4, 'Reviewed': 0.7, 'Certified': 1.0, 'Mandated': 1.3}.get(
            a.certification_tier, 1.0)

        # Persona/Zachman match
        if query_ctx.zachman_rows and a.zachman_primary_row in query_ctx.zachman_rows:
            m *= 1.15
        elif user.preferences.get('persona_view') == a.zachman_primary_row:
            m *= 1.10

        # Jurisdiction match
        jur = query_ctx.jurisdiction_context
        if jur:
            if jur in (a.applicable_jurisdictions or []):
                m *= 1.10
            elif 'global' in (a.applicable_jurisdictions or []):
                m *= 1.05

        # Recency decay
        days = (datetime.utcnow() - a.updated_at).days
        m *= max(0.7, 1.0 - days / 1000)

        hit.final_score = hit.score * m

    return sorted(hits, key=lambda h: h.final_score, reverse=True)
```

---

## 7. Dynamic facets

```python
async def compute_dynamic_facets(self, artifact_ids: list[str]) -> dict:
    body = {
        "size": 0,
        "query": {"ids": {"values": artifact_ids}},
        "aggs": {
            "togaf_domain":    {"terms": {"field": "togaf_domains",            "size": 10}},
            "certification":   {"terms": {"field": "certification_tier",        "size": 5}},
            "adm_phase":       {"terms": {"field": "adm_phases",               "size": 10}},
            "zachman_row":     {"terms": {"field": "zachman_rows",             "size": 7}},
            "technology_tag":  {"terms": {"field": "technology_tags",          "size": 15}},
            "regulatory_tag":  {"terms": {"field": "regulatory_tags",          "size": 10}},
            "jurisdiction":    {"terms": {"field": "applicable_jurisdictions", "size": 15}},
        }
    }
    r = await es.search(index="ea_hub_artifacts", body=body)
    return {
        k: [{"value": b["key"], "count": b["doc_count"]} for b in v["buckets"]]
        for k, v in r["aggregations"].items()
    }
```

---

## 8. AI synthesis

```python
async def generate_synthesis(self, query: str,
                               top_artifacts: list[Artifact]) -> AISynthesis:
    context_blocks = []
    for i, a in enumerate(top_artifacts[:5]):
        context_blocks.append(
            f"[{i+1}] {a.name} ({a.certification_tier})\n"
            f"{a.summary}\n"
            f"Trade-offs: {(a.consequences_tradeoffs or '')[:300]}\n"
        )

    system = (
        "You are the EA Knowledge Hub assistant. "
        "Synthesise architecture guidance from certified catalog content. "
        "Be direct, specific, and concise. Do not add unsupported claims. "
        "Reference sources by their [number]. Maximum 4 sentences."
    )
    user_msg = (
        f"Query: {query}\n\n"
        f"Relevant certified content:\n{''.join(context_blocks)}\n\n"
        "Synthesise a direct answer, citing relevant sources by number."
    )

    response_text = await llm_client.chat(system=system, user=user_msg, max_tokens=400)

    return AISynthesis(
        text=response_text,
        citations=[
            Citation(index=i+1, artifact_id=a.id, artifact_name=a.name)
            for i, a in enumerate(top_artifacts[:5])
        ]
    )
```

---

## 9. AI synthesis SSE streaming endpoint

```python
@router.post("/search/synthesis/stream")
async def synthesis_stream(body: SynthesisStreamRequest,
                            user: User = Depends(require_active())):
    """Stream AI synthesis token-by-token via SSE."""
    artifacts = await artifact_service.get_many(body.artifact_ids[:5])
    context = build_rag_context(artifacts)

    async def generator():
        async for chunk in llm_client.stream_chat(
            system=SYNTHESIS_SYSTEM_PROMPT,
            user=f"Query: {body.query}\n\n{context}",
            max_tokens=400
        ):
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")
```

---

## 10. Search analytics (in-app, no external systems)

```python
async def record_search_event(self, query: str, filters: dict,
                               result_count: int, user_id: str,
                               clicked_id: str = None, session_id: str = None):
    await postgres.execute("""
        INSERT INTO search_events
          (user_id, query_text, filters, result_count, no_results,
           clicked_artifact_id, session_id)
        VALUES ($1,$2,$3,$4,$5,$6,$7)
    """, user_id, query, json.dumps(filters), result_count,
         result_count == 0, clicked_id, session_id)

async def get_gap_report(self, days: int = 7) -> list[dict]:
    """Top no-results queries for steward dashboard — in-app only."""
    return await postgres.fetch("""
        SELECT query_text, count(*) AS frequency
        FROM search_events
        WHERE no_results = TRUE
          AND created_at > NOW() - ($1 || ' days')::INTERVAL
        GROUP BY query_text
        ORDER BY frequency DESC
        LIMIT 50
    """, str(days))
```

---

## 11. Typeahead / suggest endpoint

```python
@router.get("/search/suggest")
async def suggest(q: str, user: User = Depends(require_active())):
    """
    Returns up to 10 suggestions from artifact names and slugs.
    Uses Elasticsearch completion suggester or prefix query.
    Cached in Redis for 2 minutes per query prefix.
    """
    cache_key = f"suggest:{q[:50]}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    body = {
        "size": 10,
        "_source": ["id", "slug", "name", "content_type", "certification_tier"],
        "query": {
            "bool": {
                "should": [
                    {"match_phrase_prefix": {"name": {"query": q, "boost": 2}}},
                    {"match_phrase_prefix": {"slug": {"query": q}}},
                ],
                "filter": [{"terms": {"status": ["Certified","Mandated","Reviewed"]}}]
            }
        }
    }
    r = await es.search(index="ea_hub_artifacts", body=body)
    suggestions = [hit["_source"] for hit in r["hits"]["hits"]]
    await redis.setex(cache_key, 120, json.dumps(suggestions))
    return suggestions
```

---

## 12. Embedding service

```python
class EmbeddingService:

    async def embed(self, text: str) -> list[float]:
        """
        Call OpenAI-compatible embedding endpoint (internal gateway).
        Text is truncated to 8000 tokens before embedding.
        """
        client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE
        )
        response = await client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=text[:32000]  # character limit; ~8k tokens
        )
        return response.data[0].embedding

    def build_artifact_text(self, artifact: Artifact) -> str:
        """Concatenate the most semantically rich fields for embedding."""
        parts = [
            artifact.name,
            artifact.summary or '',
            getattr(artifact, 'problem_statement', '') or '',
            getattr(artifact, 'decision_statement', '') or '',
            getattr(artifact, 'consequences_tradeoffs', '') or '',
            ' '.join(artifact.technology_tags or []),
            ' '.join(artifact.regulatory_tags or []),
        ]
        return '. '.join(p for p in parts if p)
```

---

*End of SPEC-04*
