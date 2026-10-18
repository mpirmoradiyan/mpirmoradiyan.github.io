import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const notes = defineCollection({
  loader: glob({ pattern: "**/index.md", base: "./src/content/notes", generateId: ({ entry }) => entry.split("/").slice(-2, -1)[0] }),
  schema: z.object({
    title: z.string(),
    slug: z.string().optional(),
    date: z.coerce.date(),
    topic: z.string(),
    tags: z.array(z.string()).default([]),
    summary: z.string(),
    sources: z.any().optional(),
    status: z.string().optional(),
  }),
});

export const collections = { notes };
