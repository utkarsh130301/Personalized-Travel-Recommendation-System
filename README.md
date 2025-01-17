# POIBERT: A Transformer-based Model for the Tour Recommendation Problem
### Ngai Lam Ho, Kwan Hui Lim
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-BV7ZH9EX4G"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-BV7ZH9EX4G');
</script>

Tour itinerary planning and recommendation are challenging problems for tourists visiting unfamiliar cities. Many tour recommendation algorithms only consider factors such as the location and popularity of Points of Interest (POIs) but their solutions may not align well with the user's own preferences and other location constraints. Additionally, these solutions do not take into consideration of the users' preference based on their past POIs selection. In this paper, we propose POIBERT, an algorithm for recommending personalized itineraries using the BERT language model on POIs. POIBERT builds upon the highly successful BERT language model with the novel adaptation of a language model to our itinerary recommendation task, alongside an iterative approach to generate consecutive POIs.

Our recommendation algorithm is able to generate a sequence of POIs that optimizes time and users' preference in POI categories based on past trajectories from similar tourists. Our tour recommendation algorithm is modeled by adapting the itinerary recommendation problem to the sentence completion problem in natural language processing (NLP). We also innovate an iterative algorithm to generate travel itineraries that satisfies the time constraints which is most likely from past trajectories. Using a Flickr dataset of seven cities, experimental results show that our algorithm out-performs many sequence prediction algorithms based on measures in recall, precision and F1-scores.

Accepted to the 2022 IEEE International Conference on Big Data (BigData2022)

Paper: [https://arxiv.org/abs/2212.13900](https://arxiv.org/abs/2212.13900)

Citing POI-BERT
Please cite the following papers:
```
@inproceedings {10020467,
  author = {N. Ho and K. Hui Lim},
  booktitle = {2022 IEEE International Conference on Big Data (Big Data)},
  title = {POIBERT: A Transformer-based Model for the Tour Recommendation Problem},
  year = {2022},
  volume = {},
  issn = {},
  pages = {5925-5933},
  keywords = {adaptation models;urban areas;bit error rate;predictive models;prediction algorithms;natural language processing;iterative algorithms},
  doi = {10.1109/BigData55660.2022.10020467},
  url = {https://doi.ieeecomputersociety.org/10.1109/BigData55660.2022.10020467},
  publisher = {IEEE Computer Society},
  address = {Los Alamitos, CA, USA},
  month = {dec}
}
```

Source code: [https://github.com/nxh912/POIBERT_IEEE_Bigdata22](https://github.com/nxh912/POIBERT_IEEE_Bigdata22)
