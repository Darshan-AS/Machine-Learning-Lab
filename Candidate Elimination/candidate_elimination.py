"""
@author: DON
"""

import pandas


class CandidateElimination:

    def __initialize(self, x_train):
        self.__domains = pandas.DataFrame()
        for index, row in x_train.transpose().iterrows():
            self.__domains = self.__domains.append(pandas.Series(row.unique()), ignore_index=True)

        self.generic_hypothesis = pandas.DataFrame()
        self.generic_hypothesis = self.generic_hypothesis.append(
            pandas.Series(['?'] * self.__domains.shape[0]), ignore_index=True)

        self.specific_hypothesis = pandas.DataFrame()
        self.specific_hypothesis = self.specific_hypothesis.append(
            pandas.Series([None] * self.__domains.shape[0]), ignore_index=True)

    @staticmethod
    def __is_more_general(new_hypothesis, hypothesis_set):
        for index, hypothesis in hypothesis_set.iterrows():
            is_more_general = True
            for hypothesis_feature, new_hypothesis_feature in zip(hypothesis, new_hypothesis):
                is_more_general = is_more_general and (new_hypothesis_feature == '?'
                                                       or new_hypothesis_feature == hypothesis_feature)
            if not is_more_general:
                return False

        return True

    def __eliminate_generic_hypothesis(self, instance):
        for label, hypothesis in self.generic_hypothesis.iterrows():
            for hypothesis_feature, instance_feature in zip(hypothesis, instance):
                if hypothesis_feature != '?' and hypothesis_feature != instance_feature:
                    self.generic_hypothesis = self.generic_hypothesis.drop(label)
                    break

    def __generalize_specific_hypothesis(self, instance):
        for label, hypothesis in self.specific_hypothesis.iterrows():
            for (i, hypothesis_feature), instance_feature in zip(hypothesis.iteritems(), instance):
                if hypothesis_feature is None:
                    hypothesis.loc[i] = instance_feature
                elif hypothesis_feature != instance_feature:
                    hypothesis.loc[i] = '?'

            if self.__is_more_general(hypothesis, self.generic_hypothesis):
                self.generic_hypothesis = self.generic_hypothesis.drop(label)
            else:
                self.specific_hypothesis.at[label] = hypothesis

        for label, hypothesis in self.specific_hypothesis.iterrows():
            if not self.__is_more_general(hypothesis, self.specific_hypothesis):
                self.generic_hypothesis = self.generic_hypothesis.drop(label)

    def __eliminate_specific_hypothesis(self, instance):
        for label, hypothesis in self.specific_hypothesis.iterrows():
            is_consistent = False
            for hypothesis_feature, instance_feature in zip(hypothesis, instance):
                is_consistent = is_consistent and (hypothesis_feature == '?'
                                                   and hypothesis_feature == instance_feature)
            if is_consistent:
                self.specific_hypothesis.drop(labels=label)

    def __specialize_generic_hypothesis(self, instance):
        new_generic_hypothesis = pandas.DataFrame()

        for index, hypothesis in self.generic_hypothesis.iterrows():
            for (i, hypothesis_feature), instance_feature in zip(hypothesis.iteritems(), instance):
                new_hypothesis = hypothesis.copy()
                if hypothesis_feature == '?':
                    new_hypothesis.loc[i] = self.__domains.at[i, 1] \
                        if self.__domains.at[i, 0] == instance_feature else self.__domains.at[i, 0]
                elif hypothesis_feature == instance_feature:
                    new_hypothesis.loc[i] = None

                if self.__is_more_general(new_hypothesis, self.specific_hypothesis):
                    new_generic_hypothesis = new_generic_hypothesis.append(new_hypothesis, ignore_index=True)

        self.generic_hypothesis = new_generic_hypothesis

    def fit(self, x_train, y_train):
        self.__initialize(x_train)

        # print(self.specific_hypothesis, "\n\n", self.generic_hypothesis)
        for (label, row), target in zip(x_train.iterrows(), y_train):
            if target == 1:
                self.__eliminate_generic_hypothesis(row)
                self.__generalize_specific_hypothesis(row)
            elif target == 0:
                self.__eliminate_specific_hypothesis(row)
                self.__specialize_generic_hypothesis(row)

            # print('---------------------------------------------------------')
            # print(self.specific_hypothesis, "\n\n", self.generic_hypothesis)

        return self.specific_hypothesis, self.generic_hypothesis


def main():
    play_data_set = pandas.read_csv('play_data.csv', header=None)

    candidate_elimination = CandidateElimination()
    specific_hypothesis, generic_hypothesis = candidate_elimination.fit(
        play_data_set.iloc[:, :-1], play_data_set.iloc[:, -1])
    print('Specific hypothesis: \n' + specific_hypothesis.to_string(header=False, index=False))
    print()
    print('Generic hypothesis: \n' + generic_hypothesis.to_string(header=False, index=False))


if __name__ == "__main__":
    main()
