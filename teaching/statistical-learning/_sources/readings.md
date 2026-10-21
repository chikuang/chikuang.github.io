# Reading guide

Read each paper with a model–target–assumption–guarantee table beside you. The aim is to understand a small number of central arguments and reproduce an informative result, rather than to read every page of every reference. The links below point to original papers, books, or the authors' implementations.

## Weak supervision and incomplete labels

**Natarajan, N., Dhillon, I. S., Ravikumar, P. K., and Tewari, A. (2013).** [Learning with noisy labels](https://proceedings.neurips.cc/paper_files/paper/2013/hash/3871bd64012152bfb53fdf04b401193f-Abstract.html). *NeurIPS.*  
**Read for:** the relation between clean and corrupted losses under class-conditional noise. Identify the noise rates, the invertibility condition, and the consequence of estimating those rates. Used in Chapter 2.

**Elkan, C., and Noto, K. (2008).** [Learning classifiers from only positive and unlabeled data](https://cseweb.ucsd.edu/~elkan/posonly.pdf). *KDD.*  
**Read for:** the positive-label selection model and its role in relating observed labels to latent classes. Distinguish an assumption that enables a relationship from an argument that identifies an unknown constant. Used in Chapter 3.

**du Plessis, M. C., Niu, G., and Sugiyama, M. (2015).** [Convex formulation for learning from positive and unlabeled data](https://proceedings.mlr.press/v37/plessis15.html). *ICML, PMLR 37*, 1386–1394.  
**Read for:** risk rewriting and the choice of loss. Follow where the positive distribution and the feature marginal enter the objective. Chapters 3–4.

**Kiryo, R., Niu, G., du Plessis, M. C., and Sugiyama, M. (2017).** [Positive-unlabeled learning with non-negative risk estimator](https://proceedings.neurips.cc/paper_files/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html). *NeurIPS.*  
**Read for:** why an unbiased empirical component can become negative, how the correction changes the objective, and the difference between population identities and finite-sample behaviour. Chapter 4.

**Lu, N., Niu, G., Menon, A. K., and Sugiyama, M. (2019).** [On the minimal supervision for training any binary classifier from only unlabeled data](https://arxiv.org/abs/1808.10585). *ICLR.*  
**Read for:** two-mixture identification, risk rewriting, and the roles of distinct mixture priors and a target prior. Derive the two-by-two inversion yourself before reading the general formulation. Chapter 5.

### Extensions

**Bekker, J., Robberechts, P., and Davis, J. (2019).** [Beyond the selected completely at random assumption for learning from positive and unlabeled data](https://dtai.cs.kuleuven.be/research/sar). *ECML PKDD.*  
Feature-dependent positive selection requires an explicit mechanism and additional identification conditions. Use this after the introductory SCAR setting.

**Ishida, T., Niu, G., Menon, A. K., and Sugiyama, M. (2019).** [Complementary-label learning for arbitrary losses and models](https://proceedings.mlr.press/v97/ishida19a.html). *ICML, PMLR 97*, 2971–2980.  
Read the label-generation model carefully: a complementary label rules out a class, and its distribution across excluded classes matters. In a binary problem, a correct complementary label reveals the true class.

**Ratner, A. J., De Sa, C. M., Wu, S., Selsam, D., and Ré, C. (2016).** [Data programming: creating large training sets, quickly](https://proceedings.neurips.cc/paper_files/paper/2016/hash/6709e8d64a5f47269ed5cea9f625f7ab-Abstract.html). *NeurIPS.*  
Use as an introduction to combining multiple noisy supervision sources. Ask which dependencies are modelled and how source accuracy is identified.

## Preference learning and sequential decisions

**Bradley, R. A., and Terry, M. E. (1952).** [Rank analysis of incomplete block designs: I. The method of paired comparisons](https://doi.org/10.1093/biomet/39.3-4.324). *Biometrika, 39*, 324–345.  
**Read for:** the original paired-comparison model and its interpretation. Chapter 6.

**Turner, H., and Firth, D. (2012).** [Bradley–Terry models in R: the BradleyTerry2 package](https://www.jstatsoft.org/article/view/v048i09). *Journal of Statistical Software, 48*(9), 1–21.  
**Read for:** fitting, identification, model extensions, and practical interpretation. Distinguish structural graph connectivity from observed-data separation. Chapters 6–7.

**Hazan, E. (2022).** [Introduction to online convex optimization](https://arxiv.org/abs/1909.05207), second edition. MIT Press.  
**Read for:** the online protocol, static regret, projected gradient descent, and one complete regret proof. Start with the introductory and online-gradient-descent sections. Chapter 8.

**Lattimore, T., and Szepesvári, C. (2020).** [Bandit algorithms](https://tor-lattimore.com/downloads/book/book.pdf). Cambridge University Press.  
**Read for:** stochastic finite-armed bandits, confidence bounds, and the difference between cumulative reward and best-arm identification. Select the introductory stochastic-bandit and UCB material; the whole book is not required. Chapter 9.

**Yue, Y., Broder, J., Kleinberg, R., and Joachims, T. (2012).** [The k-armed dueling bandits problem](https://doi.org/10.1016/j.jcss.2011.12.028). *Journal of Computer and System Sciences, 78*(5), 1538–1556.  
**Read for:** comparative feedback and the assumptions needed to define and learn a winner. Check which regret definition the result controls. Chapter 10.

### Optional connection to generative AI

**Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. (2023).** [Direct preference optimization: your language model is secretly a reward model](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract.html). *NeurIPS.*  
Read after the paired-comparison model. Identify the policy, reference distribution, regularisation, and assumptions connecting a preference model to the optimisation objective. The course does not require language-model training.

## Inference after selection and clustering

**Lee, J. D., Sun, D. L., Sun, Y., and Taylor, J. E. (2016).** [Exact post-selection inference, with application to the lasso](https://www.stat.cmu.edu/~ryantibs/statml/lectures/Lee-Sun-Sun-Taylor.pdf). *Annals of Statistics, 44*(3), 907–927.  
**Read for:** the conditional-inference principle and a Gaussian reference distribution truncated by selection. The elementary example in Chapter 11 introduces the mechanism before the full polyhedral argument.

**Gao, L. L., Bien, J., and Witten, D. (2024).** [Selective inference for hierarchical clustering](https://arxiv.org/abs/2012.02936). *Journal of the American Statistical Association, 119*(545), 332–342; published online in 2022.  
**Read for:** the selected pair of clusters, the mean-contrast null, the selection event, and computation along a conditioned path. Use the [authors' technical guide](https://lucylgao.github.io/clusterpval/technical.html) and [`clusterpval` tutorial](https://lucylgao.github.io/clusterpval/HierCluster.html) alongside Chapter 12.

**Chen, Y. T., and Witten, D. M. (2023).** [Selective inference for k-means clustering](https://www.jmlr.org/papers/v24/22-0371.html). *Journal of Machine Learning Research, 24*(152), 1–41.  
**Read for:** the role of the algorithm's intermediate assignments and initialisation in the conditioning event. Compare the procedure with an arbitrary call to `kmeans`. Use the [authors' tutorial](https://yiqunchen.github.io/KmeansInference/articles/Tutorials.html) alongside Chapter 13.

**Neufeld, A., Dharamshi, A., Gao, L. L., and Witten, D. (2024).** [Data thinning for convolution-closed distributions](https://jmlr.org/papers/v25/23-0446.html). *Journal of Machine Learning Research, 25*(57), 1–35.  
**Optional:** study how the data model permits construction of independent pieces. Compare its assumptions with ordinary sample splitting, and identify the target retained by each approach.

## A one-page reading record

For each primary paper, record the observed data, latent quantities, target, identification assumptions, estimator or test, theorem statement, computational requirements, and a possible failure case. End with one experiment that would help you understand the result. This record can become the foundation of the [paper discussion](assignments.md#paper-discussion-and-reading-note) or [research project](project.md).
