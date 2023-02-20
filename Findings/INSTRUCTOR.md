INSTRUCTOR is a dense embedding model based on Flan-T5. It has a [website](https://instructor-embedding.github.io/).

Experiment setup: https://github.com/a0346f102085fe9f/SearchResearch/commit/f3c766c381dde410312eaf181cf295b112141fa1

These findings are not empirical as there is no performance metric other than eyeballing it.

Findings:
 - Underwhelming query performance
 - Decent similarity performance

It works, but query performance is underwhelming. It does well at finding similar documents but completely fails at search by keyword, where SPLADEv2 succeeds.

Adding or removing a period at the end of the query makes a surprisingly large amount of difference. Adding or removing the instruction for the query makes surprisingly little difference.

I tried instructor-base and instructor-large. Embedding are not interchangeable between the two, which is characterized by low similarity and no relevant results when attempting to use the smaller model to generate a query embedding and using it to score embeddings from the larger model.
