# Repository

Companion code for a Medium article.

## Business context

In 2018, a backhoe struck Williams' Transco pipeline in rural Pennsylvania. Tragically one person died and five homes were destroyed. The following investigation found that unauthorized construction had been visible in aerial imagery for six weeks before the rupture. The contractor ignored right-of-way restrictions, and the pipeline operator's monthly aerial patrol had missed the encroachment by three days.

Pipeline operators manage 3 million miles of buried infrastructure across North America, most traversing private land where construction activity is unrestricted outside the narrow right-of-way corridor. Federal regulations require aerial or satellite monitoring, but traditional methods are reactive: monthly helicopter flyovers capture snapshots, and human analysts review thousands of images looking for new structures, vegetation clearing, or earth moving equipment.

Modern computer vision transforms this workflow. Instead of humans reviewing images sequentially, self-supervised models like DINOv2 (Distillation with NO labels v2) convert each aerial tile into a 384-dimensional embedding that captures semantic content—excavators look similar in embedding space, construction sites cluster together, undisturbed forest forms a distinct distribution. When a new image appears that's distant from normal operational baselines, it flags for inspection.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).