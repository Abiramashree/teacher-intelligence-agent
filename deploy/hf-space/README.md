---
title: Teacher Intelligence Agent
emoji: 🧑‍🏫
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Teacher Intelligence Agent — hosted demo

A RAG + tool-calling agent that turns student–tutor transcripts into structured
insights, running entirely on **open-source models** (Ollama + Llama 3.2 3B) with
**no API key**.

This Space builds from the source repo:
<https://github.com/Abiramashree/teacher-intelligence-agent>

## How this Space works

- Single Docker container runs Ollama (serving `llama3.2:3b`) **and** the Streamlit app.
- The embedding model and FAISS index are baked at build time; the LLM is pulled on
  first boot, so the very first start takes a minute or two.
- Everything is local to the container — no external LLM service is called.

## Notes on the free tier

- Free CPU Spaces are always *available* but **pause after ~48h of inactivity** and
  cold-start on the next visit. For zero cold starts, upgrade the Space hardware (paid).
- CPU inference of a 3B model is usable for a demo but not instant. For snappier
  responses, set `AGENT_MODEL` / `SUMMARY_MODEL` to `llama3.2:1b` in the Space
  **Settings → Variables** (the tool-calling path is less reliable on 1B).
