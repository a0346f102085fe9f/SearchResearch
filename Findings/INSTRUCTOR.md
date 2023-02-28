INSTRUCTOR is a dense embedding model based on Flan-T5. It has a [website](https://instructor-embedding.github.io/).

| Use case | Works? |
| --- | --- |
| Find similar | Yes |
| Find by keywords | No |
| Find by description | Somewhat |

Notable:
 - Adding or removing a period at the end of the query can introduce a surprising amount of variation.
 - Exact instruction prefix doesn't seem to make that much difference.
 - Embedding are not interchangeable between the instructor-base and instructor-large, which is characterized by low similarity and no relevant results when attempting to use the smaller model to generate a query embedding and using it to score embeddings from the larger model.

