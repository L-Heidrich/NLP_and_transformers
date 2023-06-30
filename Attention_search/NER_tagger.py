import utils

texts = [
    "I packed my suitcase with clothes, toiletries, and a travel guide. The flight was smooth, and we arrived at our destination on time. The hotel had a beautiful view of the beach.",
    "Our hiking adventure was unforgettable. We explored stunning trails and encountered breathtaking views. The hiking boots and backpack were essential for the trip.",
    "The cruise ship provided a luxurious experience. We enjoyed gourmet meals, live entertainment, and relaxing spa treatments on board. The ocean view from our balcony was mesmerizing.",
    "I embarked on a cultural journey to immerse myself in the local traditions. The colorful festivals, traditional dances, and historic monuments left a lasting impression. The souvenirs I bought were a reminder of this enriching experience.",
    "The road trip was filled with excitement and spontaneous adventures. We visited charming towns, tasted delicious local cuisine, and stayed at cozy bed and breakfasts. The road atlas helped us navigate through the scenic routes.",
    "my name is lennard and I love AI"
]

seed_words = ["packed", "transported", "brought", "took", "carried", "travel",
              "trip",
              "vacation",
              "adventure",
              "explore",
              "destination",
              "hotel",
              "flight",
              "suitcase",
              "itinerary",
              "sightseeing",
              "beach",
              "hiking",
              "cruise",
              "culture",
              "road trip",
              "tour",
              "landmarks",
              "souvenirs",
              "maps"]

for text in texts:
    print(text)
    data = utils.retrieve_attention_scores(text, seed_words)
    totals = [0 for _ in range(len(data["tokens"])-1)]

    for result in data["results"]:

        scores = utils.sum_up_attention_over_heads_and_layers(result["scores"])
        totals = [a + b for a, b in zip(totals, scores)]

    utils.prettify_results(text, totals, data["tokens"])
