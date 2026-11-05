# DaniBookLM

Local, privacy-first research and book-drafting environment modeled on NotebookLM, combining active workspace memory with full-text archive retrieval.

## Language

### Workspace & Content

**Notebook (Cuaderno)**:
The bounded workspace where an author gathers sources, conducts multimodal dialogues, and synthesizes artifacts for a book or investigation.
_Avoid_: Project, workspace, app, dashboard

**Source (Fuente)**:
An immutable input material (book, paper, web page, transcript, table, image, or audio) ingested into the Notebook as primary evidence.
_Avoid_: File, attachment, upload, asset

**Collection (Colección)**:
A curated thematic bundle of Sources that can be loaded into or replacing the current Notebook in one action.
_Avoid_: Pack, preset, folder, library

**Sample (Muestra)**:
A standalone reference Source available in the catalog for immediate testing or inspection.
_Avoid_: Demo file, example, fixture

**Citation (Cita)**:
A verifiable inline reference (`[Source, Page/Section]`) linking a generated claim back to its exact origin in a Source.
_Avoid_: Link, footnote, tag

---

### Two-Tier Memory

**Desk (Mesa de Trabajo — Tier 1)**:
The subset of Sources actively checked by the author to remain in the continuous prompt prefix for immediate narrative synthesis.
_Avoid_: Active files, selected docs, short-term memory, context list

**Archive (Archivo — Tier 2)**:
The indexed repository of all Notebook Sources queried dynamically by lexical relevance to supply supporting fragments and Citations.
_Avoid_: Database, FTS table, long-term memory, vector store

---

### Studio & Synthesis

**Studio**:
The synthesis workspace where the author transforms Desk and Archive knowledge into structured outputs and persistent notes.
_Avoid_: Right panel, tools sidebar, export menu

**Studio Artifact (Artefacto de Studio)**:
A structured analytical synthesis generated from active Sources—specifically a Study Guide, Visual Mind Map, Flashcards & Quiz, Concept Glossary, Timeline, or Key Quotes compilation.
_Avoid_: Report, summary, quick action, output, audit

**Concept Map / Visual Mind Map (Mapa Conceptual / Visual)**:
A Studio Artifact that maps theoretical relationships, clusters, and entity bridges across active Sources as an interactive visual graph with exact Citations.
_Avoid_: Dictionary, index, gap audit, word list, wireframe

**Flashcards & Quiz (Tarjetas de Estudio)**:
A Studio Artifact that compiles high-impact study cards with question prompts, revealed answers, and key takeaways backed by Citations.
_Avoid_: Exam, test suite, survey, questionnaire

**Note (Nota)**:
A mutable piece of authorial writing or pinned synthesis saved permanently within the Studio.
_Avoid_: Comment, memo, scratchpad, post

**Deep Research**:
An autonomous web investigation that compiles external findings into a cited dossier and imports it as a new Source.
_Avoid_: Web search, scraper, crawler
