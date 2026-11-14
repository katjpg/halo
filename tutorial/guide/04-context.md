# Context and persistence

Scientific synthesis brings together publications, molecular results, and previous research decisions. Retrieval selects the records relevant to a question; saved decisions preserve guidance for later work. HALO builds a separate context for each task so agents receive the records needed for that request.

## Concepts and records

| Concept or record | Purpose |
| --- | --- |
| Context | The selected molecules, results, evidence, and decisions supplied for a task. |
| Evidence passage | A publication excerpt with a citation ID and a relationship to the requested molecule. |
| Numerical result | A calculated, predicted, or measured value with its own ID, unit, and method. |
| Decision | Saved design guidance or an accepted comparison and its reason. |

### Build context for a task

1. **Resolve the scope.** Identify the molecule or group of molecules that the request concerns. Design starts from the active parent.
2. **Select records.** The `Optimizer` receives the parent's profile, requested changes, relevant edits, and decisions. The `Researcher` receives passages and selected numerical results, each with distinct citation IDs.
3. **Apply the instructions.** The agent receives the request, selected records, and constraints for its task. [Agent workflows](03-agents.md)

### Retrieve scientific evidence

HALO searches ChEMBL and PubChem for the requested structure and its analogs. It combines database IDs for matching canonical structures and ranks analogs by Morgan Tanimoto similarity. Papers and patents use the same selected analogs.

| Evidence group | Relationship |
| --- | --- |
| Exact | A ChEMBL activity or PubChem compound record links the publication to the requested structure. |
| Analog | The publication is linked to a similar structure, recorded with its fingerprint similarity. |
| Topic | The publication matches the query, without a verified link to the structure. |

When the query specifies a target or assay, matching records must satisfy those requirements before ranking. A ChEMBL activity must match them within the same record. A document found through PubChem or a topic search must match them in its title or one abstract passage. Relevant papers and patents are then selected from the matching records.

A missing result means that no match was found in the records checked. A patent excerpt leaves claim coverage unreviewed. Failed lookups are reported separately.

### Retrieve full-text passages

For a question about a structural change and property, the `Researcher` plans a focused search:

1. Find open-access papers indexed as CC BY.
2. Confirm a CC BY link in each article's XML permissions.
3. Extract complete sentences that match the change, property, and any requested target or assay.
4. Rank and deduplicate the passages, then select whole quotes for synthesis.

Tables, figures, supplements, and references are excluded. Full-text passages retain their citation IDs, publication identifiers, license URLs, and paragraph and sentence locations. A verified publication link is required to classify a passage as exact or analog evidence; otherwise, it remains topic evidence.

Phrase matching identifies relevant passages to inspect. It does not establish that the edit caused the property change. [Passage fields and retrieval limits](05-reference.md#research-passages)

## Tutorial

Use a project opened as shown in the [project tutorial](../../examples/basic-agent.md).

### 1. Research a design question

Name the structural change and property in the query.

```python
response = project.research(
    "Find papers that report how fluorination changes aqueous solubility."
)
print(response)
```

The output separates exact, analog, and topic sources. It also reports excluded publications and failed lookups.

### 2. Inspect source records

Select exact evidence when you need publications linked to the requested structure.

```python
for passage in response.sources.for_scope("exact"):
    print(passage.title, passage.url)
```

Use `response.sources.exclusions` for excluded publication IDs and reasons, or `response.sources.unavailable` for failed lookups. These records are saved with `response.turn`.

### 3. Record and recall decisions

Save guidance in natural language, then ask about it later.

```python
project.remember("Retain the aryl fluoride in every round.")
reply = project.ask("What design guidance did I save for future rounds?")
print(reply)
```

Guidance applies to the next batch that uses it and saves candidates. Wording such as “in every round” makes it ongoing. Recall returns current decisions by default; the recall action also supports decision IDs and an `as_of` cutoff.

### 4. Continue in a new session

Reopen the same project to start a new conversation. Its saved decisions remain available. Use `resume=True` to continue the latest session instead.

```python
project = halo.projects.get(project.id)
reply = project.ask("What design guidance applies to this project?")
print(reply)
```

### 5. Compare saved measurements

```python
reply = project.ask("Summarize the saved assay results for the active parent.")
print(reply)
```

Measurements can be compared when the assay, endpoint, method, unit, and conditions match. Bounds retain their inequalities. Findings cite publication passages through `evidence_ids` and numerical results through `result_ids`.

## Set up the database

For a fresh Supabase database running PostgreSQL 15 or later, apply `schema.sql`. Set `HALO_DATABASE_URL` to the connection string from the project's Connect dialog, then run this command from the repository root. [Supabase connection instructions](https://supabase.com/docs/guides/database/connecting-to-postgres)

```bash
psql "$HALO_DATABASE_URL" -v ON_ERROR_STOP=1 --single-transaction -f src/halo/db/sql/schema.sql
```

| Environment variable | Purpose |
| --- | --- |
| `SUPABASE_URL` | Supabase project URL. |
| `SUPABASE_PUBLISHABLE_KEY` | Caller authentication and data access. |
| `SUPABASE_SECRET_KEY` | Server-side write operations. |
| `SUPABASE_AUTH_ISSUER` | Optional token issuer override. |

Keep `SUPABASE_SECRET_KEY` on the server. Projects, runs, sessions, assays, and decisions are saved in the project database. [HTTP API](05-reference.md#http-api)
