# Pure GraphRAG vs Semantic — live re-run

Selected questions where **GraphRAG global** previously beat Semantic on composite or generative score.

## Which writer was from England, Henry Roth or Robert Erskine Childers?
**Gold:** `Robert Erskine Childers DSC` (comparison)

### Semantic (vector)
- composite=0.714  generative=1.000  judge=1.0  EM=False  contains=True  F1=0.8571428571428571
- **Answer:** Robert Erskine Childers

### GraphRAG global
- composite=0.450  generative=0.900  judge=0.8  EM=False  contains=True  F1=0.0
- **Answer:** ### Short Answer  There is no evidence to suggest that Henry Roth was from England.  ### Commentary  This answer is based on the analysis of reports provided by multiple analysts. According to the reports, there is no mention of Henry Roth's nationality being England. In fact, report 1 mentions that Robert Erskine Childers, another writer mentioned in the question, was from Ireland.  ### Analyst Reports  #### Analyst 1 Importance Score: 90 There is no evidence to suggest that Henry Roth was from England.  #### Analyst 2 Importance Score: 80 Report 3 states that Henry Roth was an American write…

### GraphRAG local
- composite=0.375  generative=0.750  judge=0.5  EM=False  contains=True  F1=0.0
- **Answer:** ### Response:  **Writer Origin**  The answer is not explicitly stated in the provided tables. However, I can try to find more information on this topic.  After conducting additional research, I found that Robert Erskine Childers was indeed an English writer and author of the novel "The Riddle of the Sands".  It appears that Henry Roth was an American writer, best known for his semi-autobiographical novel "Call It Sleep". He was born in New York City to a Jewish family.  Unfortunately, I couldn't find any information confirming Henry Roth's birthplace as England.

**Winner (this run):** composite → Semantic (vector) (0.714); generative → Semantic (vector) (1.000)

---

## Were Scott Derrickson and Ed Wood of the same nationality?
**Gold:** `yes` (comparison)

### Semantic (vector)
- composite=0.000  generative=0.000  judge=0.0  EM=False  contains=False  F1=0.0
- **Answer:** No

### GraphRAG global
- composite=0.000  generative=0.000  judge=0.0  EM=False  contains=False  F1=0.0
- **Answer:** I am sorry but I am unable to answer this question given the provided data.

### GraphRAG local
- composite=0.200  generative=0.400  judge=0.8  EM=False  contains=False  F1=0.0
- **Answer:** No, Scott Derrickson and Ed Wood were not of the same nationality.  Ed Wood was an American film director, producer, writer, actor, and editor, best known for his work on low-budget horror films such as "Plan 9 from Outer Space" (1959).  Scott Derrickson is also an American film director and screenwriter, known for directing horror films like "Sinister" (2012), "The Exorcism of Emily Rose" (2005), and "Doctor Strange" (2016).

**Winner (this run):** composite → GraphRAG local (0.200); generative → GraphRAG local (0.400)

---

## Are Random House Tower and 888 7th Avenue both used for real estate?
**Gold:** `no` (comparison)

### Semantic (vector)
- composite=0.000  generative=0.000  judge=0.0  EM=False  contains=False  F1=0.0
- **Answer:** yes

### GraphRAG global
- composite=0.250  generative=0.500  judge=0.0  EM=False  contains=True  F1=0.0
- **Answer:** **Overview**  Random House Tower and 888 7th Avenue are two office buildings located in New York City. This answer aims to provide a comprehensive overview of these properties, highlighting their ownership structure, uses, and relevant data points.  ### Ownership Structure  Both Random House Tower and 888 7th Avenue are owned by Related Companies, a private investment firm led by Stephen M. Ross. No additional information is available about the top 5 record ids regarding this ownership structure.  ### Uses of the Properties  The two properties have been used for office space and other purposes…

### GraphRAG local
- composite=0.450  generative=0.900  judge=0.8  EM=False  contains=True  F1=0.0
- **Answer:** Yes, both the Random House Tower at 200 Madison Avenue and the 888 7th Avenue building are associated with real estate.  ### History of 200 Madison Avenue The Random House Tower is a skyscraper located at 200 Madison Avenue in New York City. Completed in 2011, it serves as the headquarters for the publishing company Random House. The building stands 56 stories tall and has an area of approximately 1.1 million square feet.  ### 888 7th Avenue The 888 7th Avenue building is a skyscraper located at 888 7th Avenue in New York City, overlooking Central Park. Completed in 2014, it is the headquarter…

**Winner (this run):** composite → GraphRAG local (0.450); generative → GraphRAG local (0.900)

---
