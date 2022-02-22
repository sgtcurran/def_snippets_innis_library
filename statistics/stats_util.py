import scipy
import pandas as pd

## PROBABILITY DISTRIBUTION FUNCTIONS

probability_distribution = scipy.stats._distn_infrastructure.rv_frozen

def generate_random_value(distribution: probability_distribution, size = 1):
    '''
    Return a single random value, or array of random values, using the given distribution.

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
    chi2, p, degf, expected = scipy.stats.chi2_contingency(observed)
    
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
    if p < alpha:
        print('\nReject H0')
    else: 
        print('\nFail to Reject H0')