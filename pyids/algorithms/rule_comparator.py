from abc import abstractmethod, ABC

class RuleComparator(ABC):

    @abstractmethod
    def greater_than(self, rule1, rule2):
        pass

    @abstractmethod
    def lesser_than(self, rule1, rule2):
        pass


class RuleComparatorF1(ABC):

    @abstractmethod
    def greater_than(self, rule1, rule2):
        """
        precedence operator. Determines if this rule
        has higher precedence. Rules are sorted according
        to their f1 score.
        """

        f1_score_self = rule1.calc_f1()
        f1_score_other = rule2.calc_f1()


        return f1_score_self > f1_score_other

    @abstractmethod
    def lesser_than(self, rule1, rule2):
        """
        rule precedence operator
        """
        return not rule1 > rule2


class RuleComparatorCBA():
    pass