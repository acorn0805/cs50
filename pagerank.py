import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])#corpus is a dictionary 
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    n = len(corpus)
    distribution = dict()

    # 如果当前页面有出链
    if corpus[page]:
        for p in corpus:
            distribution[p] = (1 - damping_factor) / n
        for linked_page in corpus[page]:
            distribution[linked_page] += damping_factor / len(corpus[page])
    else:
        # 如果当前页面没有出链，则认为它指向所有页面（包括自己）
        for p in corpus:
            distribution[p] = 1 / n

    return distribution
  
    


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    dict={page: 0 for page in corpus}# Initialize PageRank values to 0
    page=random.choice(list(corpus.keys()))# Start with a random page
    for _ in range(n):
        dict[page] += 1
        model = transition_model(corpus, page, damping_factor)
        pages = list(model.keys())
        probabilities = list(model.values())
        page = random.choices(pages, weights=probabilities, k=1)[0]
    return {page: value / n for page, value in dict.items()}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    n = len(corpus)
    pagerank = {page: 1 / n for page in corpus}  # Initialize PageRank values to 1/N
    new_pagerank = pagerank.copy()
    converged = False

    while not converged:
        converged = True
        for page in corpus:
            total = 0
            for possible_page in corpus:
                if page in corpus[possible_page]:
                    total += pagerank[possible_page] / len(corpus[possible_page])
                if not corpus[possible_page]:  # If a page has no links, treat it as linking to all pages
                    total += pagerank[possible_page] / n
            new_value = (1 - damping_factor) / n + damping_factor * total
            if abs(new_value - pagerank[page]) > 0.001:
                converged = False
            new_pagerank[page] = new_value
        pagerank = new_pagerank.copy()

    return pagerank

if __name__ == "__main__":
    main()
