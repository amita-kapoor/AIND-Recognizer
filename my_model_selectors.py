import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores

        best_score, best_model = float("inf"), None

        ## For each Gaussian model calculate BIC values, return the best model based on BIC values
        # Lower the BIC value better the model
        logN = np.log(self.X.shape[0])
        for num_comp in range(self.min_n_components, self.max_n_components+1):
            try:
                model = GaussianHMM(n_components=num_comp, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)  # Obtain the log-likelihood of the trained model
                num_features = model.n_features #self.X.shape[1]
                p = num_comp * num_comp  + 2 * num_features * num_comp - 1
                #p = num_comp * (num_comp - 1) + 2 * num_features * num_comp
                #logN = np.log(self.X.shape[0])
                bic = -2 * logL + p * logN
                if bic < best_score:
                    best_score, best_model = bic, model
            except:
                #print('Exception occurred')  # Print an error message and continue to next fitting
                continue

        if best_model is not None:
            return best_model
        return self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    models, values = {}, {}
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        if len(SelectorDIC.models) == 0:  # If there are no models we generate models for each word for all num_component in the specified range
            self.generate_dictionary(self)

        best_score, best_model = float("-inf"), None

        for num_comp in range(self.min_n_components, self.max_n_components + 1):
            models, ml = SelectorDIC.models[num_comp], SelectorDIC.values[num_comp]  # Now models contain the GMM model for the specified No.
            # of components and ml contains the loglikelihood value for the same number of components

            if (self.this_word not in ml):
                continue

            avg = np.mean([ml[word] for word in ml.keys() if word != self.this_word])  # 1/(M-1)SUM(log(P(X(all but i))
            dic = ml[self.this_word] - avg   # DIC = LogL - avg => model fit to 'this_word'  - the average of the likelihood on all other words

            if dic > best_score:
                best_score, best_model = dic, models[self.this_word]

        if best_model is not None:
            return best_model
        return self.base_model(self.n_constant)


    def generate_dictionary(cls, inst):
        for num_comp in range(inst.min_n_components, inst.max_n_components + 1):
            num_comp_models, num_comp_ml = {}, {}

            for word in inst.words.keys():
                X, lengths = inst.hwords[word]
                try:
                    model = GaussianHMM(n_components=num_comp, covariance_type="diag", n_iter=1000,
                                        random_state=inst.random_state, verbose=False).fit(X, lengths)
                    logL = model.score(X, lengths)
                    num_comp_models[word] = model  # GMM model of this 'word'
                    num_comp_ml[word] = logL     # Loglikelihood of this word
                except:
                    #print('Exception occurred')  # Print an error message and continue to next fitting
                    continue

            ### For Each value of num_component now SelectorDIC.models contain model of each word
            # And SelectorDIC.values contain the logliklihood for it.
            SelectorDIC.models[num_comp] = num_comp_models
            SelectorDIC.values[num_comp] = num_comp_ml


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV

        best_score, best_model = float("-inf"), None

        for num_comp in range(self.min_n_components, self.max_n_components + 1):
            scores, n_splits = [], 3
            model, logL = None, None

            if (len(self.sequences) < n_splits):
                break

            split_method = KFold(random_state=self.random_state, n_splits=n_splits)  # Spilt data into 3

            ## For each Gaussian model trained calculate average log likelihood values
            # The one with highest average is best
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                try:
                    model = GaussianHMM(n_components=num_comp, covariance_type="diag", n_iter=1000,
                                        random_state= self.random_state, verbose=False).fit(X_train, lengths_train)
                    logL = model.score(X_test, lengths_test)
                    scores.append(logL)
                except:
                    #print('Exception occurred')  # Print an error message and continue to next fitting
                    continue

            avg = np.average(scores) if len(scores) > 0 else float("-inf")

            if avg > best_score:
                best_score, best_model = avg, model

        if best_model is not None:
            return best_model
        return self.base_model(self.n_constant)
