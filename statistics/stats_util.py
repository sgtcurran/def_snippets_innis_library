import scipy

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