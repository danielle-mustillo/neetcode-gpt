from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        corpus = list(corpus)
        merges = []

        for _ in range(0, num_merges):

            # count the occurences
            counts = defaultdict(int)
            for pnt2 in range(1, len(corpus)):
                pnt1 = pnt2 - 1
                l1 = corpus[pnt1]
                l2 = corpus[pnt2]

                counts[(l1, l2)] += 1

            # find max pair
            max_chars = []
            max_found = -1
            for char_pair, count in counts.items():
                if count > max_found:
                    max_chars = [char_pair]
                    max_found = count
                elif count == max_found:
                    max_chars.append(char_pair)
            
            # sort to find the lexicographically first item
            max_chars.sort()
            first, second = max_chars[0]
            merges.append([first, second])

            # merge back
            new_corpus = []
            removed_idxs = []
            for i in range(0, len(corpus) - 1):
                # print(pnt1, pnt2, corpus)
                if corpus[i] == first and corpus[i + 1] == second:
                    corpus[i] = first + second
                    corpus[i+1] = ""
                    removed_idxs.append(i+1)
            
            # remove pairs
            removed_idxs.reverse()
            for idx in removed_idxs:
                del corpus[idx]

            # corpus = new_corpus
            # print(corpus)

        return merges
        pass
