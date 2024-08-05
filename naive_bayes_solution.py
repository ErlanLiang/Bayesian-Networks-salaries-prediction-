from bnetbase import Variable, Factor, BN
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
    sum = 0
    for value in factor.values:
        sum += value
    if sum == 0:
        new_factor.values = factor.values
        return new_factor
    for i in range(len(new_factor.values)):
        new_factor.values[i] = factor.values[i] / sum
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
        new_assignment = [v for v in assignment if v not in variable.domain()]   
        if tuple(new_assignment) not in dic:
            dic[tuple(new_assignment)] = [len(variable.domain()) - 1, factor.get_value(assignment)]

        if dic[tuple(new_assignment)][0] > 1:
            dic[tuple(new_assignment)][0] -= 1
            dic[tuple(new_assignment)][1] += factor.get_value(assignment)
        else:
            new_factor.add_values([new_assignment + [dic[tuple(new_assignment)][1] + factor.get_value(assignment)]])
    return new_factor

def multiply(factor_list: list[Factor]):
    '''
    Multiply a list of factors together.
    Do not modify any of the input factors. 

    :param factor_list: a list of Factor objects.
    :return: a new Factor object resulting from multiplying all the factors in factor_list.
    '''
    while len(factor_list) > 1:
        # print("==============================================")

        cur_factor1 = factor_list.pop(0)
        cur_factor2 = factor_list.pop(0)
        # cur_factor1.recursive_print_values(cur_factor1.get_scope())
        # print(" ")
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # cur_factor2.recursive_print_values(cur_factor2.get_scope())
        # print(" ")
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        new_scope = []
        for factor in [cur_factor1, cur_factor2]:
            for var in factor.get_scope():
                if var not in new_scope:
                    new_scope.append(var)
        new_factor = Factor("Result", new_scope)
        
        for assignment in itertools.product(*[v.domain() for v in new_scope]):
            # print("assignment: ", assignment)
            # find the which variables in the assignment are in the original factors

            factor1_get = []
            factor2_get = []
            
            for scope in cur_factor1.get_scope():
                for v in assignment:
                    if v in scope.domain():
                        # print("scope: "+ scope.name + " domain: " + str(scope.domain()))
                        # print("v: ", v)
                        factor1_get.append(v)
            for scope in cur_factor2.get_scope():
                for v in assignment:
                    if v in scope.domain():
                        factor2_get.append(v)
            # print("factor1_get: ", factor1_get)
            # print("factor1 scope: ", cur_factor1.get_scope())
            # print("factor2_get: ", factor2_get)
            # print("factor2 scope: ", cur_factor2.get_scope())
            new_factor.add_values([list(assignment) + [cur_factor1.get_value(factor1_get) * cur_factor2.get_value(factor2_get)]])
        new_factor = normalize(new_factor)
        factor_list.append(new_factor)
        # new_factor.recursive_print_values(new_factor.get_scope())
        # print(" ")
        # print(" ")
        # print(" ")
    # factor_list[0].recursive_print_values(factor_list[0].get_scope())
    return factor_list[0]  

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
    # copy the factors
    factors = bayes_net.factors()

    for i in range(len(factors)):
        factor = factors.pop(0)
        for evidence in EvidenceVars:
            if evidence in factor.get_scope():
                # print("evidence: ", evidence.get_evidence())
                factor = restrict(factor, evidence, evidence.get_evidence())
        factors.append(factor)
    
    # print(" ")
    # print(" ")
    # for factor in factors:
    #     print("factor: ", factor.name)
    #     factor.recursive_print_values(factor.get_scope())

    # multiply the factors
    factor = multiply(factors)

    # print(" ")
    # print(" ")
    # print(factor.get_scope())
    # factor.recursive_print_values(factor.get_scope())

    # sum out the variables
    count = 0
    for var in bayes_net.variables():
        if var != var_query:
            factor = sum_out(factor, var)

            count += 1
            print("count: ", count)
            print(" ")
            print(" ")

            # if count == 1:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            #     print("var: ", var.name)
            #     factor.recursive_print_values(factor.get_scope())

    # normalize the factor
    factor = normalize(factor)

    return factor




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
    E1 = ["Work", "Occupation", "Education", "Relationship"]
    E2 = ["Work", "Occupation", "Education", "Relationship", "Gender"]

    # Step 1: Set up evidence for E1 and E2
    E1_vars = [bayes_net.get_variable(var) for var in E1]
    E2_vars = [bayes_net.get_variable(var) for var in E2]

    E1_combinations = itertools.product(*[var.domain() for var in E1_vars])
    # print("E1_combinations: ", E1_combinations)
    E2_combinations = itertools.product(*[var.domain() for var in E2_vars])

    # Step 2: Perform variable elimination for E1 and E2
    factor_E1 = ve(bayes_net, bayes_net.get_variable("Salary"), E1_vars)
    factor_E2 = ve(bayes_net, bayes_net.get_variable("Salary"), E2_vars)

    

    # Step 3: Calculate the percentages based on the question
    if question == 1:
        total_count = 0
        strict_greater_count = 0
        
        for assignment_E1 in E1_combinations:
            print("assignment_E1: ", assignment_E1)
            for i, var in enumerate(E1_vars):
                var.set_evidence(assignment_E1[i])
            
            factor_E1 = ve(bayes_net, bayes_net.get_variable("Salary"), E1_vars)
            
            for assignment_E2 in E2_combinations:
                print("assignment_E2: ", assignment_E2)
                for i, var in enumerate(E2_vars):
                    var.set_evidence(assignment_E2[i])
                
                factor_E2 = ve(bayes_net, bayes_net.get_variable("Salary"), E2_vars)
                
                # print(factor_E1.get_scope())
                # print(factor_E1.get_scope()[0].domain())
                # # print([assignment_E1] + [">=$50K"])
                prob_E1 = factor_E1.get_value([">=50K"])
                print("prob_E1: ", prob_E1)
                prob_E2 = factor_E2.get_value([">=50K"])
                print("prob_E2: ", prob_E2)
                
                if prob_E1 > prob_E2:
                    strict_greater_count += 1
                total_count += 1
        
        return (strict_greater_count / total_count) * 100
    
    elif question == 2:
        return 0.0
    elif question == 3:
        return 0.0
    elif question == 4:
        return 0.0
    elif question == 5:
        return 0.0
    elif question == 6:
        return 0.0
    else:
        return 0.0
        
        
        


if __name__ == '__main__':
    nb = naive_bayes_model('data/adult-train.csv')
    # for i in range(1,7):
    #     print("explore(nb,{}) = {}".format(i, explore(nb, i)))
    nb.get_variable("Work").set_evidence("Government")
    nb.get_variable("Occupation").set_evidence("Professional")
    nb.get_variable("Education").set_evidence("Bachelors")
    nb.get_variable("Relationship").set_evidence("Husband")
    nb.get_variable("Gender").set_evidence("Male")
    factor = ve(nb, nb.get_variable("Salary"), [nb.get_variable("Work"), nb.get_variable("Occupation"), nb.get_variable("Education"), nb.get_variable("Relationship"), nb.get_variable("Gender")])
    factor.recursive_print_values(factor.get_scope())
