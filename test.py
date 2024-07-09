history = (
    {
        "query": "cybersecurity",
        "results": [
            {"link": "https://en.wikipedia.org/wiki/Cybersecurity", "frequency": 2},
            {"link": "https://www.ibm.com/security/cybersecurity", "frequency": 1},
        ],
        "total_frequency": 3,
    },
    {
        "query": "virtual reality",
        "results": [
            {"link": "https://en.wikipedia.org/wiki/Virtual_reality", "frequency": 3},
            {"link": "https://www.ibm.com/blogs/virtual-reality", "frequency": 2},
        ],
        "total_frequency": 5,
    }
)


# get the links of the query called "cybersecurity"

def get_links(query):
    for h in history:
        if h["query"] == query:
            return [r["link"] for r in h["results"]]
