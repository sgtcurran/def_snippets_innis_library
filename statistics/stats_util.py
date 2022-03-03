from scipy import stats
import pandas as pd

## PROBABILITY DISTRIBUTION FUNCTIONS

probability_distribution = stats._distn_infrastructure.rv_frozen

def generate_random_value(distribution: probability_distribution = stats.randint(0, 2), size: int = None):
    '''
    Return a single random value, or array of random values, using the given distribution. By
    default will generate randomly a 0 or 1.

    This function utilizes the rvs method of the distribution object. The purpose of this
    function is simply to provide a more meaningful name to the rvs function. This function
    in no way changes the behavior of the rvs function.
    '''

    return distribution.rvs(size)

def prob_of_value_discrete(distribution: probability_distribution, value: int) -> float:
    '''
    Returns the probability that the distribution will randomly generate the given value, given
    that the distribution is a discrete distribution.

    This function utilizes the pmf method of the distribution object. The purpose of this
    function is simply to provide a more meaningful name to the pmf function. This function
    in no way changes the behavior of the pmf function.
    '''

    return distribution.pmf(value)

def prob_of_value_continuous(distribution: probability_distribution, value: float) -> float:
    '''
    Returns the probability that the distribution will randomly generate the given value, given
    that the distribution is a continuous distribution.

    This function utilizes the pdf method of the distribution object. The purpose of this
    function is simply to provide a more meaningful name to the pdf function. This function
    in no way changes the behavior of the pdf function.
    '''

    return distribution.pdf(value)

def prob_less_than_value(distribution: probability_distribution, value: float) -> float:
    '''
    Returns the probability that the distribution will randomly generate a value less than
    or equal to the given value.

    This function utilizes the cdf method of the distribution object. The purpose of this
    function is simply to provide a more meaningful name to the cdf function. This function
    in no way changes the behavior of the cdf function.
    '''

    return distribution.cdf(value)

def value_less_than_prob(distribution: probability_distribution, probability: float) -> float:
    '''
    Given the probability of generating a random value less than or equal to some value n,
    returns the value n.

    This function utilizes the ppf method of the distribution object. The purpose of this
    function is simply to provide a more meaningful name to the ppf function. This function
    in no way changes the behavior of the ppf function.
    '''

    return distribution.ppf(probability)

def prob_greater_than_value(distribution: probability_distribution, value: float) -> float:
    '''
    Returns the probability that the distribution will randomly generate a value greater than
    the given value.

    This function utilizes the sf method of the distribution object. The purpose of this
    function is simply to provide a more meaningful name to the sf function. This function
    in no way changes the behavior of the sf function.
    '''

    return distribution.sf(value)

def value_greater_than_prob(distribution: probability_distribution, probability: float) -> float:
    '''
    Given the probability of generating a random value greater than some value n, returns
    the value n.

    This function utilizes the isf method of the distribution object. The purpose of this
    function is simply to provide a more meaningful name to the isf function. This function
    in no way changes the behavior of the isf function.
    '''

    return distribution.isf(probability)
    
def evaluate_hypothesis(p: float, alpha: float = 0.05, output: bool = True) -> bool:
    '''
    Compare the p value to the established alpha value to determine if the null hypothesis
    should be rejected or not.
    '''

    if p < alpha:
        if output:
            print('\nReject H0')
        return False
    else: 
        if output:
            print('\nFail to Reject H0')
        return True

## CHI-SQUARED TEST FUNCTION

def chi2_test(data_for_category1, data_for_category2, alpha=.05):

    '''
    Given two subgroups from a dataset, conducts a chi-squared test for independence and outputs 
    the relevant information to the console. 

    Utilizes the method provided in the Codeup curriculum for conducting chi-squared test using
    scipy and pandas. 
    '''
    
    # create dataframe of observed values
    observed = pd.crosstab(data_for_category1, data_for_category2)
    
    # conduct test using scipy.stats.chi2_contingency() test
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    
    # round the expected values
    expected = expected.round(1)
    
    # output
    print('Observed\n')
    print(observed.values)
    print('---\nExpected\n')
    print(expected)
    print('---\n')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    
    # evaluate the hypothesis against the established alpha value
    evaluate_hypothesis(p, alpha)

# PEARSONR CORRELATION TEST FUNCTION

def correlation_test(data_for_category1, data_for_category2, alpha = 0.05):
    '''
    Given two subgroups from a dataset, conducts a correlation test for linear relationship and outputs 
    the relevant information to the console. 

    Utilizes the method provided in the Codeup curriculum for conducting correlation test using
    scipy and pandas. 
    '''

    # conduct test using scipy.stats.peasonr() test
    r, p = stats.pearsonr(data_for_category1, data_for_category2)

    # output
    print(f'r = {r:.4f}')
    print(f'p = {p:.4f}')

    # evaluate the hypothesis against the established alpha value
    evaluate_hypothesis(p, alpha)

def equal_var_test(*args, alpha: float = 0.05) -> bool:
    '''
    Given two or more subgroups from a dataset, conducts a test of equal variance and returns whether or
    not p is less than alpha.
    '''

    f, p = stats.levene(*args)
    return evaluate_hypothesis(p, alpha, output = False)

def central_limit_theorem_test(*args, n_clt: int = 30) -> bool:
    '''
    Given two or more subgroups from a dataset, determines whether or not we have large enough sample
    sizes to use the central limit theorem to assume normal distribution.
    '''

    sample_sizes = [arg.size for arg in args]
    return min(sample_sizes) >= n_clt

def two_sample_ttest(
    sample1: pd.core.series.Series,
    sample2: pd.core.series.Series,
    alpha: float = 0.05,
    n_clt: int = 30,
    alternative: str = 'two-sided'
) -> None:
    '''
    Given two independent samples from a dataset, conducts a two sample t-test to compare means and outputs
    the relevant information to the console.

    Parameters
    ----------
    sample1 : Pandas Series
        A pandas series containing an independent sample from a dataset. This sample should be independent
        from sample2.

    sample2 : Pandas Series
        A pandas series containing an independent sample from a dataset. This sample should be independent
        from sample1.

    alpha : float, default 0.05
        The alpha value (derived from the confidence level) to use when determining whether or not to
        reject the null hypothesis.

    n_clt : int, default 3
        The minimum sample size required to use a parametric test. This parameter is used to determine
        if the central limit theorem can be used to assume a normal distribution of data. If the sample
        sizes are less than n_clt a non-parametric test will be used.

    alternative : str, default 'two-sided'
        The type of two sample t-test to perform. Possible values are 'two-sided', 'less', or 'greater',
        where 'less' and 'greater' are one tail t-tests and 'two-sided' is a two tail t-test.

    Returns
    -------
    None : Nothing is returned by this function. All relevant information is printed to the console.

    Examples
    --------
    >>> import stats_util as su

    >>> su.two_sample_ttest(sample1, sample2)
    >>> su.two_sample_ttest(sample1, sample2, alpha = 0.01, n_clt = 50)
    >>> su.two_sample_ttest(sample1, sample2, alternative = 'less')
    '''

    # Are the samples large enough to assume normal distribution?
    normal_dist = central_limit_theorem_test(sample1, sample2, n_clt = n_clt)
    print(f'Samples contain more than {n_clt} observations: {normal_dist}')

    # if our samples are normally distributed use a parametric test
    if normal_dist:
        # Do the subgroups have equal variance?
        equal_var = equal_var_test(sample1, sample2, alpha = alpha)
        print(f'Samples have equal variances: {equal_var}')
        print(f'Using parametric test...')
        f, p = stats.ttest_ind(sample1, sample2, equal_var = equal_var, alternative = alternative)

    # otherwise use a non-parametric test
    else:
        print(f'Using non-parametric test...')
        f, p = stats.mannwhitneyu(sample1, sample2, alternative = alternative)

    evaluate_hypothesis(p, alpha)

def anova_test(*args, alpha: float = 0.05, n_clt: int = 30) -> None:
    '''
    Given three or more subgroups from a dataset, conducts an ANOVA test to compare means and outputs
    the relevant information to the console.

    Parameters
    ----------
    *args : Pandas Series
        Any number of Pandas series. This function is intended to perform a test on 3 or more datasets
        so 3 or more series should be passed as arguments.

    alpha : float, default 0.05
        The alpha value (derived from the confidence level) to use when determining whether or not to
        reject the null hypothesis.

    n_clt : int, default 3
        The minimum sample size required to use a parametric test. This parameter is used to determine
        if the central limit theorem can be used to assume a normal distribution of data. If the sample
        sizes are less than n_clt a non-parametric test will be used.

    Returns
    -------
    None : Nothing is returned by this function. All relevant information is printed to the console.

    Examples
    --------
    >>> import stats_util as su

    >>> su.anova_test(sample1, sample2, sample3)
    >>> su.anova_test(sample1, sample2, sample3, alpha = 0.01, n_clt = 50)
    '''

    # Are the samples large enough to assume normal distribution?
    normal_dist = central_limit_theorem_test(*args, n_clt = n_clt)
    print(f'Samples contain more than {n_clt} observations: {normal_dist}')

    # if our samples are normally distributed use a parametric test
    if normal_dist:
        # Do the subgroups have equal variance?
        equal_var = equal_var_test(*args, alpha = alpha)
        print(f'Samples have equal variances: {equal_var}')
        print(f'Using parametric test...')
        f, p = stats.f_oneway(*args, equal_var = equal_var)

    # otherwise use a non-parametric test
    else:
        print(f'Using non-parametric test...')
        f, p = stats.kruskal(*args)

    evaluate_hypothesis(p, alpha)