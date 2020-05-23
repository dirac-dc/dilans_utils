from __future__ import annotations
from functools import partial
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

from sparse_dot_topn.sparse_dot_topn import sparse_dot_topn
from sparse_dot_topn.sparse_dot_topn_threaded import sparse_dot_topn_threaded


class TFIDFMatcher:
    def __init__(self, low_bound: float, top_n: int, vectoriser: TfidfVectorizer, n_jobs: int = 1):
        """Make TFIDF-weighted cosine similarities from arbitrary TfidfVectorizer
        Non-abstract base class that has to be given an instantiated TfidfVectorizer. Its children here don't.
        Parameters
        ----------
        low_bound : float, [0., 1.]
            Lowest value of tf-idf match to retain in the trained tf-idf matrix. Comparisons not found in the matrix
            will subsequently be set to 0.
        top_n : int
            Maximum number of results to calculate per rw of the left `pd.Series` given to `transform`. Values are
            calculated in descending order of priority.
        vectoriser : TfidfVectorizer
            The string vectoriser with `.fit(X)` and `.transform(X)` methods available.
            IMPORTANT: Must output normed vectors from its transform method, i.e. ||v|| = 1.0.
        n_jobs : int > 0, default 1
            If n_jobs > 1, the parallel implementation of sparse_dot_topn will be called with n_jobs setting the number
            of threads.
        """
        self.vectoriser = vectoriser
        self.top_n = top_n
        self.low_bound = low_bound
        self.n_jobs = n_jobs

        if n_jobs == 1:
            self.sparse_dot_topn = partial(sparse_dot_topn, ntop=top_n, lower_bound=low_bound)
        else:
            self.sparse_dot_topn = partial(sparse_dot_topn_threaded, ntop=top_n, lower_bound=low_bound, n_jobs=n_jobs)

    def fit(self, text_series: pd.Series) -> TFIDFMatcher:
        """Fit underlying vectoriser."""
        self.vectoriser.fit(text_series)
        return self

    def transform(self, text_series_1: pd.Series, text_series_2: Optional[pd.Series] = None, idf: bool = False
                  ) -> pd.DataFrame:
        """Create `pd.DataFrame` of `top_n` matches for text inputs.
        Parameters
        ----------
        text_series_1 : pd.Series
            Series of text
        text_series_2 : pd.Series
            Series of text
        idf : bool
            Add extra column to returned pd.DataFrame which gives the sum of IDF scores for the left hand side strings
            (from `text_series_1`). Note that this will vary depending on the instantiation params of the underlying
            `TfidfVectorizer` - for example, the setting `smooth_idf` affects this.
            Will cause `AttributeError` if `TfidfVectorizer` was instantiated with `use_idf=False`.
            NOTE: Although this can be used to discriminate between company names in OpenCorporates, another approach
            might be better:
            - Just clean the names and do values counts
            - Take the average IDF instead of the sum IDF.
            - If we do want to use summation, doesn't it make more sense to sum the DFs then invert them?
        Returns
        -------
        pd.DataFrame
            Table with columns 'left' and 'right' which give the names of each entry, and 'similarity' which give the
            TF-IDF-weighted cosine similarity score.
        """
        deduplicated_input_1 = text_series_1.drop_duplicates()
        vector = self.vectoriser.transform(deduplicated_input_1.values)

        if text_series_2 is None:
            deduplicated_input_2 = deduplicated_input_1
            vector_2 = vector
        else:
            deduplicated_input_2 = text_series_2.drop_duplicates()
            vector_2 = self.vectoriser.transform(deduplicated_input_2.values)

        if not vector.nnz or not vector_2.nnz:
            # Workaround for sparse_dot_topn crash when one of the vectors has no nonzero entries:
            df_out = pd.DataFrame({
                'left': np.array([], dtype=object),
                'right': np.array([], dtype=object),
                'similarity': np.array([], dtype=float)
            })
        else:
            matches = self.top_matches(vector, vector_2)
            df_out = self.matches_to_df(matches, text_series_1=deduplicated_input_1, text_series_2=deduplicated_input_2)

        if idf:
            d_idfs = dict(zip(deduplicated_input_1.values, (vector > 0) * self.vectoriser.idf_))
            df_out['idf'] = df_out['left'].map(d_idfs)

        return df_out

    def top_matches(self, vectors_a, vectors_b):
        """Calculates cosine distance between tf-idf vectors."""
        vectors_a = vectors_a.tocsr()
        vectors_b = vectors_b.transpose().tocsr()
        M = vectors_a.shape[0]
        N = vectors_b.shape[1]

        idx_dtype = np.int32
        nnz_max = M * self.top_n

        indptr = np.zeros(M + 1, dtype=idx_dtype)
        indices = np.zeros(nnz_max, dtype=idx_dtype)
        data = np.zeros(nnz_max, dtype=vectors_a.dtype)

        # Cython glue which populates the above numpy arrays by reference.
        self.sparse_dot_topn(
            n_row=M,
            n_col=N,
            a_indptr=np.asarray(vectors_a.indptr, dtype=idx_dtype),
            a_indices=np.asarray(vectors_a.indices, dtype=idx_dtype),
            a_data=vectors_a.data,
            b_indptr=np.asarray(vectors_b.indptr, dtype=idx_dtype),
            b_indices=np.asarray(vectors_b.indices, dtype=idx_dtype),
            b_data=vectors_b.data,
            c_indptr=indptr,
            c_indices=indices,
            c_data=data
        )

        return csr_matrix((data, indices, indptr), shape=(M, N))

    @staticmethod
    def matches_to_df(sparse_matrix, text_series_1, text_series_2) -> pd.DataFrame:
        """Extract a meaningful `pd.DataFrame` from the matches matrix for full_cosine_pipeline"""
        non_zeros = sparse_matrix.nonzero()

        return pd.DataFrame({'left': text_series_1.values[non_zeros[0]],
                             'right': text_series_2.values[non_zeros[1]],
                             'similarity': sparse_matrix.data.round(10)})


class TFIDFWordMatcher(TFIDFMatcher):
    def __init__(self, low_bound=0., top_n=10, min_word_length=1, n_jobs=1):
        """Make TFIDF-weighted cosine similarities.
        Convenience class that builds its own TfidfVectorizer for decomposing strings at the word level.
         Parameters
         ----------
        low_bound : float, [0., 1.]
            Lowest value of tf-idf match to retain in the trained tf-idf matrix. Comparisons not found in the matrix
            will subsequently be set to 0.
        top_n : int
            Maximum number of results to calculate per row of the left `pd.Series` given to `transform`. Values are
            calculated in descending order of priority.
        min_word_length : int
            Minimum number of characters consisting a word for TFIDF purposes.
        n_jobs : int > 0, default 1
            If n_jobs > 1, the parallel implementation of sparse_dot_topn will be called with n_jobs setting the number
            of threads.
         """
        token_pattern = r'\b' + r'\w' * min_word_length + r'+\b'
        super().__init__(low_bound, top_n, TfidfVectorizer(analyzer='word', token_pattern=token_pattern), n_jobs=n_jobs)


class TFIDFCharacterMatcher(TFIDFMatcher):
    def __init__(self, low_bound=0., top_n=10, ngram_range=(1, 1), n_jobs=1):
        """Make TFIDF-weighted cosine similarities at character-level.
        Convenience class that builds its own TfidfVectorizer for decomposing strings at the character level.
        Parameters
        ----------
        low_bound : float, [0., 1.]
            Lowest value of tf-idf match to retain in the trained tf-idf matrix. Comparisons not found in the matrix
            will subsequently be set to 0.
        top_n : int
            Maximum number of results to calculate per row of the left `pd.Series` given to `transform`. Values are
            calculated in descending order of priority.
        ngram_range : tuple of Int
            Passed to the TfidfVectorizer: The lower and upper boundary of the range of n-values for different n-grams
            to be extracted. All values of n such that min_n <= n <= max_n will be used.
         """
        super().__init__(low_bound, top_n, TfidfVectorizer(analyzer='char', ngram_range=ngram_range), n_jobs=n_jobs)o
