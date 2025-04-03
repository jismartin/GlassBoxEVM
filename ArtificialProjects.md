# Artificial project data

The [MT dataset](https://www.projectmanagement.ugent.be/research/project_scheduling/rcpsp) was originally created to test the algorithms described in the book *Measuring Time* [1] and has since been used in several studies. In this work, we use a subset of 900 Activity-on-the-Node (AoN) networks, each consisting of 30 activities with randomly assigned durations and costs. The dataset used here is located in the folder `dataset/set1SP`. The topological structure of these networks is determined by the SP-factor, which ranges from 0.1 to 0.9 in increments of 0.1 (100 networks are generated for each SP value).

The SP-factor, introduced by Vanhoucke along with other indicators of topological complexity [2], serves to estimate the difficulty of a project scheduling problem. As the original dataset does not include these indicators for all samples, we compute them in this work.

> References:
> 1. Vanhoucke, M. (2009). Measuring time: Improving project performance using earned value management (Vol. 136). Springer Science & Business Media.
> 2. Vanhoucke, M., Coelho, J., Debels, D., & Tavares, L. (2004). On the morphological structure of a network.
