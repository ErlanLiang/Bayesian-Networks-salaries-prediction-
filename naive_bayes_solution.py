from bnetbase import Variable, Factor, BN
from time import time
import csv
import itertools


def normalize(factor: Factor) -> Factor:
    '''
    Normalize the factor such that its values sum to 1.
    Do not modify the input factor.

    :param factor: a Factor object. 
    :return: a new Factor object resulting from normalizing factor.
    '''
    # copy the factor
    new_factor = Factor(factor.name + "_normalized", factor.get_scope())
    sum_total = sum(factor.values)
    if sum_total == 0:
        new_factor.values = factor.values
        return new_factor
    for i in range(len(new_factor.values)):
        new_factor.values[i] = factor.values[i] / sum_total
    return new_factor


def restrict(factor: Factor, variable: Variable, value):
    '''
    Restrict a factor by assigning value to variable.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to restrict.
    :param value: the value to restrict the variable to
    :return: a new Factor object resulting from restricting variable to value.
             This new factor no longer has variable in it.

    '''
    new_factor = Factor(factor.name, factor.get_scope())

    for assignment in itertools.product(*[v.domain() for v in factor.get_scope()]):
        # print("assignment: ", assignment)
        if value in assignment:
            # print("value in: ", value)
            new_factor.add_values([[v for v in assignment] + [factor.get_value(assignment)]])

    # print("new_factor: ", new_factor.values)
    # print("new_factor: ", new_factor.get_scope())
    return new_factor

def sum_out(factor: Factor, variable: Variable):
    '''
    Sum out a variable variable from factor factor.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to sum out.
    :return: a new Factor object resulting from summing out variable from the factor.
             This new factor no longer has variable in it.
    '''
    new_scope = [v for v in factor.get_scope() if v != variable]
    new_factor = Factor(factor.name, new_scope)

    dic = {}
    for assignment in itertools.product(*[v.domain() for v in factor.get_scope()]):
        # remove the input variable from the assignment
        # print("=====================================")
        # print("dic:", dic)
        # print("assignment: ", assignment)
        new_assignment = [v for v in assignment if v not in variable.domain()]   
        # print("new_assignment: ", new_assignment)
        if tuple(new_assignment) not in dic:
            dic[tuple(new_assignment)] = [len(variable.domain()) - 1, factor.get_value(assignment)]

        elif dic[tuple(new_assignment)][0] > 1:
            dic[tuple(new_assignment)][0] -= 1
            dic[tuple(new_assignment)][1] += factor.get_value(assignment)
        else:
            dic[tuple(new_assignment)][1] += factor.get_value(assignment)
            # new_factor.add_values([new_assignment + [dic[tuple(new_assignment)][1] + factor.get_value(assignment)]])
    
    for key, value in dic.items():
        new_factor.add_values([list(key) + [value[1]]])
    return new_factor

def multiply(factor_list: list[Factor]):
    '''
    Multiply a list of factors together.
    Do not modify any of the input factors. 

    :param factor_list: a list of Factor objects.
    :return: a new Factor object resulting from multiplying all the factors in factor_list.
    '''
    # while len(factor_list) > 1:
    #     cur_factor1 = factor_list.pop(0)
    #     cur_factor2 = factor_list.pop(0)
    if len(factor_list) == 0:
        return None
    
    new_scope = set()
    for factor in factor_list:
        new_scope.update(factor.get_scope())
    new_scope = list(new_scope)
    
    new_factor = Factor("Result", new_scope)
    
    # print("new_scope: ", new_scope)
    for assignment in itertools.product(*[v.domain() for v in new_scope]):
        # print("assignment: ", assignment)
        # find the which variables in the assignment are in the original factors
        cur_value = 1
        for factor in factor_list:
            # print("factor: ", factor.name)
            factor_get = []
            for scope in factor.get_scope():
                index = new_scope.index(scope)
                value = assignment[index]
                factor_get.append(value)
            # print("factor_get: ", factor_get)
            cur_value *= factor.get_value(factor_get)
            # print("cur_value: ", cur_value)
        new_factor.add_values([list(assignment) + [cur_value]])
    return new_factor
        

    #         factor1_get = []
    #         factor2_get = []
            
    #         for scope in cur_factor1.get_scope():
    #             for v in assignment:
    #                 if v in scope.domain():
    #                     factor1_get.append(v)

    #         for scope in cur_factor2.get_scope():
    #             for v in assignment:
    #                 if v in scope.domain():
    #                     factor2_get.append(v)

    #         new_factor.add_values([list(assignment) + [cur_factor1.get_value(factor1_get) * cur_factor2.get_value(factor2_get)]])
    #     # new_factor = normalize(new_factor)
    #     factor_list.append(new_factor)
    # return factor_list[0]  

def ve(bayes_net: BN, var_query: Variable, EvidenceVars: list[Variable]):
    '''

    Execute the variable elimination algorithm on the Bayesian network bayes_net
    to compute a distribution over the values of var_query given the 
    evidence provided by EvidenceVars. 

    :param bayes_net: a BN object.
    :param var_query: the query variable. we want to compute a distribution
                     over the values of the query variable.
    :param EvidenceVars: the evidence variables. Each evidence variable has 
                         its evidence set to a value from its domain 
                         using set_evidence.
    :return: a Factor object representing a distribution over the values
             of var_query. that is a list of numbers, one for every value
             in var_query's domain. These numbers sum to 1. The i-th number
             is the probability that var_query is equal to its i-th value given 
             the settings of the evidence variables.

    For example, assume that
        var_query = A with Dom[A] = ['a', 'b', 'c'], 
        EvidenceVars = [B, C], and 
        we have called B.set_evidence(1) and C.set_evidence('c'), 
    then VE would return a list of three numbers, e.g. [0.5, 0.24, 0.26]. 
    These numbers would mean that 
        Pr(A='a'|B=1, C='c') = 0.5, 
        Pr(A='a'|B=1, C='c') = 0.24, and 
        Pr(A='a'|B=1, C='c') = 0.26.

    '''
    # lst = ['Male', 'North-America', 'Private', 'Bachelors', '>=50K', 'Not-Married', 'Professional', 'Not-in-family', 'White']
    # lst = ['Not-in-family', 'White', 'Male', 'North-America', 'Private', 'Bachelors', '>=50K', 'Not-Married', 'Professional']
    # lst = ['Professional', 'Not-in-family', 'White', 'Male', 'North-America', 'Private', 'Bachelors', '>=50K', 'Not-Married']
    # dict = {
    #     "Gender": "Male",
    #     "Country": "North-America",
    #     "Work": "Private",
    #     "Education": "Bachelors",
    #     "Salary": ">=50K",
    #     "MaritalStatus": "Not-Married",
    #     "Occupation": "Professional",
    #     "Relationship": "Not-in-family",
    #     "Race": "White"
    # }
    # copy the factors
    factors = bayes_net.factors()

    start = time()
    for i in range(len(factors)):
        factor = factors.pop(0)
        for evidence in EvidenceVars:
            if evidence in factor.get_scope():
                # print("evidence: ", evidence.get_evidence())
                factor = restrict(factor, evidence, evidence.get_evidence())
        factors.append(factor)
    
    # for factor in factors:
    #     scope = factor.get_scope()
    #     cur_vars = []
    #     for var in scope:
    #         cur_vars.append(dict[var.name])
    #     print("cur_vars: ", cur_vars)
    #     print("factor: ", factor.name)
    #     print("factor: ", factor.get_value(cur_vars))
    #     print("=====================================")
    


    # print("time restrict: ", time() - start)
    # start = time()


    # Eliminate hidden variables
    hidden_vars = [v for v in bayes_net.variables() if v.name != var_query.name]
    # print("hidden: ", hidden_vars)

    for hidden_var in hidden_vars:
        # print("hidden_var: ", hidden_var)
        cur_factors = []
        new_factor = None
        for factor in factors:
            if hidden_var in factor.get_scope():
                # print("got factor: ", factor.name)
                cur_factors.append(factor)
        if cur_factors:
            new_factor = sum_out(multiply(cur_factors), hidden_var)
        updated_factors = []
        # Add factors which don't have hidden variable
        for f in factors:
            if f not in cur_factors:
                updated_factors.append(f)
        # Add factors which do have hidden variable(already summed out)
        if new_factor:
            updated_factors.append(new_factor)
        # Add back
        factors = updated_factors

    if len(factors) > 1:
        result_factor = multiply(factors)
    else:
        result_factor = factors[0]
    return normalize(result_factor)
        
            


    # # multiply the factors
    # factor = multiply(factors)
    # # get the value out and see if it is correct
    # # factor.recursive_print_values(factor.get_scope())
    # # print("scope: ", factor.get_scope())
    # # print("got_value: ", factor.get_value(lst))
    # # print("got_value: ", factor.get_value(['White', 'Female', 'North-America', 'Private', 'Bachelors', '>=50K', 'Not-Married', 'Professional', 'Not-in-family']))

    # # sum out the variables
    
    # for var in bayes_net.variables():
    #     if var != var_query:
    #         # lst.remove(var.get_evidence())
    #         factor = sum_out(factor, var)
    #         # print("scope: ", factor.get_scope())
    #         # print("got_value at sum"+ var.name + ": ", factor.get_value(lst))

    # # normalize the factor
    # factor = normalize(factor)
    # print("time normalize: ", time() - start)

    # return factor
    

def naive_bayes_model(data_file, variable_domains = {"Work": ['Not Working', 'Government', 'Private', 'Self-emp'], "Education": ['<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors', 'Masters', 'Doctorate'], "Occupation": ['Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service', 'Professional'], "MaritalStatus": ['Not-Married', 'Married', 'Separated', 'Widowed'], "Relationship": ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'], "Race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'], "Gender": ['Male', 'Female'], "Country": ['North-America', 'South-America', 'Europe', 'Asia', 'Middle-East', 'Carribean'], "Salary": ['<50K', '>=50K']}, class_var = Variable("Salary", ['<50K', '>=50K'])):
    '''
   NaiveBayesModel returns a BN that is a Naive Bayes model that 
   represents the joint distribution of value assignments to 
   variables in the Adult Dataset from UCI.  Remember a Naive Bayes model
   assumes P(X1, X2,.... XN, Class) can be represented as 
   P(X1|Class)*P(X2|Class)* .... *P(XN|Class)*P(Class).
   When you generated your Bayes bayes_net, assume that the values 
   in the SALARY column of the dataset are the CLASS that we want to predict.
   @return a BN that is a Naive Bayes model and which represents the Adult Dataset. 
    '''
    ### READ IN THE DATA
    input_data = []
    with open(data_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None) #skip header row
        for row in reader:
            input_data.append(row)

    ### DOMAIN INFORMATION REFLECTS ORDER OF COLUMNS IN THE DATA SET
    #variable_domains = {
    #"Work": ['Not Working', 'Government', 'Private', 'Self-emp'],
    #"Education": ['<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors', 'Masters', 'Doctorate'],
    #"Occupation": ['Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service', 'Professional'],
    #"MaritalStatus": ['Not-Married', 'Married', 'Separated', 'Widowed'],
    #"Relationship": ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
    #"Race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
    #"Gender": ['Male', 'Female'],
    #"Country": ['North-America', 'South-America', 'Europe', 'Asia', 'Middle-East', 'Carribean'],
    #"Salary": ['<50K', '>=50K']
    #}
    

    ### CREATE VARIABLES
    variables = []
    for header in headers:
        variables.append(Variable(header, variable_domains[header]))
    
    variables[1].dom.remove('Professional')
    variables[1].dom.append('Edu_Professional')
    # print("variables: ", variables)
    # print("variables[1].domain(): ", variables[1].domain())
    

    ### CREATE FACTORS
    factors = []
    for i, variable in enumerate(variables):
        if variable.name != class_var.name:
            factors.append(Factor(variable.name +"|" +class_var.name, [variable, variables[-1]]))
        else:
            factors.append(Factor(variable.name, [variable]))
    # print("factors: ", factors)

    ### COUNT OCCURRENCES
    # Set values for factors based on data
    # crate dictionary，key is variable name，values are variable's dict{domain1 : [domain1 with salary >=50K times, domain1 with salary <50K times],
    #  domain2 : [domain2 with salary >=50K times, domain2 with salary <50K times], ...}

    # initialize the dictionary
    dic = {}
    for i in variables:
        if i.name == "Education":
            # since there are two Professional in the data, we need to change the name of the Education to Edu_Professional
            dic[i.name] = {}
            for domain in i.domain():
                if domain == "Professional":
                    dic[i.name]["Edu_Professional"] = [0, 0]
                else:
                    dic[i.name][domain] = [0, 0]
        elif i.name != class_var.name:
            dic[i.name] = {}
            for domain in i.domain():
                dic[i.name][domain] = [0, 0]
        else:
            dic[i.name] = {}
            for domain in i.domain():
                dic[i.name][domain] = 0
    # print("dic: ", dic)

    # count the occurrences
    for row in input_data:
        for i, value in enumerate(row):
            # print("value: ", value)
            # print("variables[i].name: ", variables[i].name)
            if i != 8:
                if row[8] == '>=50K':
                    if value == 'Professional' and variables[i].name == "Education":
                        dic[variables[i].name]["Edu_Professional"][0] += 1
                    else:
                        dic[variables[i].name][value][0] += 1
                else:
                    if value == 'Professional' and variables[i].name == "Education":
                        dic[variables[i].name]["Edu_Professional"][1] += 1
                    else:
                        dic[variables[i].name][value][1] += 1
            else:
                    dic[variables[i].name][value] += 1
    # print("dic: ", dic)

    # calculate the probability and add to the factor
    for i, factor in enumerate(factors):
        values = []
        # get all the assignments for this factor
        for assignment in itertools.product(*[v.domain() for v in factor.get_scope()]):
            # print("assignment: ", assignment)
            # get the value of the assignment
            if len(assignment) == 2:
                if assignment[1] == '>=50K':
                    values.append(list(assignment) + [dic[factor.get_scope()[0].name][assignment[0]][0] / dic["Salary"][">=50K"]])
                else:
                    values.append(list(assignment) + [dic[factor.get_scope()[0].name][assignment[0]][1] / dic["Salary"]["<50K"]])
            else:
                values.append(list(assignment) + [dic["Salary"][assignment[0]] / (dic["Salary"]["<50K"] + dic["Salary"][">=50K"])])
        # print("values: ", values)
        factor.add_values(values)

    ### CREATE BAYES NET
    bn = BN("BN", variables, factors)

    # # print the factors
    # for factor in factors:
    #     print("factor: ", factor.name)
    #     factor.recursive_print_values(factor.get_scope())

    return bn


def explore(bayes_net: BN, question: int) -> float:
    '''    Input: bayes_net---a BN object (a Bayes bayes_net)
           question---an integer indicating the question in HW4 to be calculated. Options are:
           1. What percentage of the women in the data set end up with a P(S=">=$50K"|E1) that is strictly greater than P(S=">=$50K"|E2)?
           2. What percentage of the men in the data set end up with a P(S=">=$50K"|E1) that is strictly greater than P(S=">=$50K"|E2)?
           3. What percentage of the women in the data set with P(S=">=$50K"|E1) > 0.5 actually have a salary over $50K?
           4. What percentage of the men in the data set with P(S=">=$50K"|E1) > 0.5 actually have a salary over $50K?
           5. What percentage of the women in the data set are assigned a P(Salary=">=$50K"|E1) > 0.5, overall?
           6. What percentage of the men in the data set are assigned a P(Salary=">=$50K"|E1) > 0.5, overall?
           @return a percentage (between 0 and 100)
           core evidence set (E1) using the values assigned to the following variables: [Work, Occupation, Education, and Relationship Status]
           extended evidence set (E2) using the values assigned to the following variables: [Work, Occupation, Education, Relationship Status, and Gender]
    ''' 
    test_data = []
    with open('data/adult-test.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip header
        for row in reader:
            test_data.append(row)

    salary_variable = bayes_net.get_variable("Salary")
    numerator, denominator = 0, 0

    for row in test_data:
        gender = row[6]
        actual_salary = row[8]

        # Core evidence set E1
        E1 = {"Work": row[0], "Occupation": row[3], "Education": row[1],
              "Relationship": row[4]}
        if E1["Education"] == "Professional":
            E1["Education"] = "Edu_Professional"

        # Set evidence for E1
        E1_vars = []
        for var_name, value in E1.items():
            var = bayes_net.get_variable(var_name)
            var.set_evidence(value)
            E1_vars.append(var)

        # Extended evidence set E2
        E2 = {**E1, "Gender": gender}  # Merge E1 and Gender to create E2

        # Set evidence for E2
        E2_vars = []
        for var_name, value in E2.items():
            var = bayes_net.get_variable(var_name)
            var.set_evidence(value)
            E2_vars.append(var)

        # Compute probabilities using variable elimination
        prob_E1 = ve(bayes_net, salary_variable, E1_vars).get_value(['>=50K'])
        
            
        prob_E2 = ve(bayes_net, salary_variable, E2_vars).get_value(['>=50K'])

        # Process each question
        if question == 1:
            if gender == "Female":
                denominator += 1
                if prob_E1 > prob_E2:
                    numerator += 1
        elif question == 2:
            if gender == "Male":
                denominator += 1
                if prob_E1 > prob_E2:
                    numerator += 1
        elif question == 3:
            if gender == "Female" and prob_E1 > 0.5:
                denominator += 1
                if actual_salary == ">=50K":
                    numerator += 1
        elif question == 4:
            if gender == "Male" and prob_E1 > 0.5:
                denominator += 1
                if actual_salary == ">=50K":
                    numerator += 1
        elif question == 5:
            if gender == "Female":
                denominator += 1
                if prob_E1 > 0.5:
                    numerator += 1
        elif question == 6:
            if gender == "Male":
                denominator += 1
                if prob_E1 > 0.5:
                    numerator += 1

        # Reset evidence for next iteration
        for var in E1_vars + E2_vars:
            var.set_evidence(var.domain()[0])  # Reset to the first domain value

    if denominator == 0:
        return 0

    print("numerator: ", numerator)
    print("denominator: ", denominator)
    return (numerator / denominator) * 100
        
        
        


if __name__ == '__main__':
    # start = time()
    # nb = naive_bayes_model('data/adult-train.csv')
    # # print("time model build: ", time() - start)
    # # print("explore(nb,{}) = {}".format(1, explore(nb, 1)))
    # for i in range(1,7):
    #     print("=====================================")
    #     print("explore(nb,{}) = {}".format(i, explore(nb, i)))
    # # nb.get_variable("Work").set_evidence("Private")
    # # nb.get_variable("Occupation").set_evidence("Professional")
    # # nb.get_variable("Education").set_evidence("Bachelors")
    # # nb.get_variable("Relationship").set_evidence("Not-in-family")
    # # nb.get_variable("Gender").set_evidence("Male")
    # # factor = ve(nb, nb.get_variable("Salary"), [nb.get_variable("Work"), nb.get_variable("Occupation"), nb.get_variable("Education"), nb.get_variable("Relationship"), nb.get_variable("Gender")])
    # # factor.recursive_print_values(factor.get_scope())
    # # print("time: ", time() - start)


    #  E,B,S,W,G example
    E, B, S, G, W = Variable('E', ['e', '-e']), Variable('B', ['b', '-b']), Variable('S', ['s', '-s']), Variable('G', ['g', '-g']), Variable('W', ['w', '-w'])
    FE, FB, FS, FG, FW = Factor('P(E)', [E]), Factor('P(B)', [B]), Factor('P(S|E,B)', [S, E, B]), Factor('P(G|S)', [G,S]), Factor('P(W|S)', [W,S])


    FE.add_values([['e',0.1], ['-e', 0.9]])
    FB.add_values([['b', 0.1], ['-b', 0.9]])
    FS.add_values([['s', 'e', 'b', .9], ['s', 'e', '-b', .2], ['s', '-e', 'b', .8],['s', '-e', '-b', 0],
                    ['-s', 'e', 'b', .1], ['-s', 'e', '-b', .8], ['-s', '-e', 'b', .2],['-s', '-e', '-b', 1]])
    FG.add_values([['g', 's', 0.5], ['g', '-s', 0], ['-g', 's', 0.5], ['-g', '-s', 1]])
    FW.add_values([['w', 's', 0.8], ['w', '-s', .2], ['-w', 's', 0.2], ['-w', '-s', 0.8]])

    # try restrict FS with S = s
    print("restrict FS with S = s")
    new_factor = restrict(FS, S, '-b')
    new_factor.recursive_print_values(new_factor.get_scope())
    for scope in new_factor.get_scope():
        print("scope: ", scope.name)
        print("domain: ", scope.domain())   