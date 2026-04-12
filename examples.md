# Example GraphRAG Responses

Sample questions run against the live Neo4j Aura graph via `python scripts/graphrag_app.py`.

---

## Q1: What is the typical customer journey before a purchase event?
**Retriever:** VectorCypher

The typical customer journey before a purchase event often includes several stages, which can vary based on the marketing channels used. Based on the provided context, here are the common stages observed:

1. **Awareness**: Customers may first become aware of the product through various channels such as organic social media, webinars, or display ads. For example, CUST-0021 and CUST-0040 started their journey with trial starts via organic social.

2. **Engagement**: After awareness, customers engage with the product, often through trials or clicks. For instance, CUST-0031 engaged with a trial start via paid search, while CUST-0064 clicked via display.

3. **Consideration**: Customers evaluate the product, which may involve participating in webinars or further interactions. CUST-0079 participated in a webinar and subsequently made a purchase.

4. **Conversion**: Finally, customers make a purchase decision, which is the culmination of their journey. For example, CUST-0087 made a purchase via email after engaging with the product through clicks and trials.

Overall, the journey typically involves awareness, engagement, consideration, and conversion, with various marketing channels facilitating each stage.

---

## Q2: What products did customers who booked a demo end up purchasing?
**Retriever:** VectorCypher

The products that customers who booked a demo ended up purchasing are:

SKU-0025, SKU-0011, SKU-0018, SKU-0014, SKU-0003, SKU-0027, SKU-0014, SKU-0026, SKU-0014, SKU-0008

---

## Q3: Which agencies drove the most revenue from Enterprise customers?
**Retriever:** Text2Cypher

The agency that drove the most revenue from Enterprise customers is **AGY-06**, with a total revenue of **$30,510.16**.

---

## Q4: Which campaigns have the best return on spend?
**Retriever:** Text2Cypher

The campaigns with the best return on spend are:

| Campaign | ROI |
|---|---|
| CAMP-025 | 858.82x |
| CAMP-004 | 706.68x |
| CAMP-017 | 691.75x |

---

## Q5: Which products have the highest revenue in the APAC region?
**Retriever:** Text2Cypher

The product with the highest revenue in the APAC region is **SKU-0014**, with a total revenue of **$34,713.83**.
