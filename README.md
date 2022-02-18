## Welcome to the super awesomeness ultra Uber underground library of functions, code snippets, and how to pubs. 

## I if you have some super awesomeness aka “Da Sauce” please feel free to the library.

## Format:
# Please place in appropriate subject and if it applies to two or more please state that.

## What does it do? 
# Use common terms when writing your function, code snippets, and how to. 
# example

# data cleaning applies to and any dataset 

def clean_currency(x): # def name what does it do
    """ If the value is a string, then remove currency symbol and delimiters
    otherwise, the value is numeric and can be converted
    """
    if isinstance(x, str):
        return(x.replace('$', '').replace(',', ''))
    return(x)

