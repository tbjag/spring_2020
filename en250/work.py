# coding: utf-8
import random
import time
import csv
POPULATION_DENSITY = .75
SOCIAL_DIST_PERCENTAGE = .5
INITIAL_SICK_PERCENTAGE = 0.001
#PERSON_MAX_AGE = 28470

class Person:
    def __init__(self, alive, social_distance, sick, age, sick_days):
        self.alive = alive
        self.social_distance = social_distance
        self.sick = sick
        self.age = age
        self.sick_days = 0
        self.reinfected = 0


environment = [[None for _ in range(200)] for _ in range(100)]
dead_from_sickness = 0
healed = 0
social_dis_deaths = 0
not_social_dis_deaths = 0
age_deaths = 0

def initializeEnvironment():
    global POPULATION_DENSITY
    global SOCIAL_DIST_PERCENTAGE
    global INITIAL_SICK_PERCENTAGE
    for i in range(len(environment)):
        for j in range(len(environment[i])):
            x = None
            exist_chance = random.random()
            dice = random.random()
            sick_chance = random.random()
            if random.random() < POPULATION_DENSITY: #percent chance there is a person in this location
                 x = Person(True, (dice < SOCIAL_DIST_PERCENTAGE), (sick_chance < INITIAL_SICK_PERCENTAGE), random.randint(1, 100), 0)
            environment[i][j] = x

#simulates the passing of time and interaction between people
def step():
    for i in range(len(environment)):
        for j in range(len(environment[i])):
            person = environment[i][j]
            if person != None:
                #The person dies if they are too old or have been sick for too long
                if determineDeath(person):
                    environment[i][j] = None
                    break

                #Log people that are sick and give them a small chance of healing
                determineHealth(person)
                surrounding_people = get_surrounding(i, j)
                determineInfectionRate(person, surrounding_people)
                personMove(i, j)

#The chance that someone will die depending on how many days they've been sick already. Each index is one day
def deathByAge(dice, prob, person):
    global dead_from_sickness
    global social_dis_deaths
    global not_social_dis_deaths
    if dice < prob:
        dead_from_sickness+=1
        if person.social_distance:
            social_dis_deaths+=1
        else:
            not_social_dis_deaths+=1
        return True
    return False


#returns True if the person dies of old age
def determineDeath(person):
    global dead_from_sickness
    global social_dis_deaths
    global not_social_dis_deaths

    if person != None: #assume
        if person.sick:
            dice = random.random()
            if(person.age < 10): #age 0-9 no fatality
                return False
            elif person.age < 40: #age 10-39 .2% fatality
                return deathByAge(dice, .000071429, person)
            elif person.age < 50: #40 - 49
                return deathByAge(dice, .0001429, person)
            elif person.age < 60: #50 - 59
                return deathByAge(dice, .0004643, person)
            elif person.age < 70: #60 - 69
                return deathByAge(dice, .0012857, person)
            elif person.age < 80: #70 - 79
                return deathByAge(dice, .0028571, person)
            else: # 80+ 
                return deathByAge(dice, .005, person)
    return False


def determineHealth(person):
    global healed
    if person.sick:
        person.sick_days+=1
    if person.sick_days > 28:
        person.sick_days = 0
        person.sick = False 
        healed += 1
    

#Changes the persons sick status based on the condition of the people around them
def determineInfectionRate(person, surrounding_people):
    if person.reinfected and person.sick < 28:
        person.reinfected += 1
        return
    else:
        person.reinfected = 0
    sick_count = 0
    for neighbor in surrounding_people:
        if neighbor != None:
            if neighbor.sick:
                sick_count+=1

    if person.social_distance:
        infection_rate = sick_count * 0.0001
    else:
        infection_rate = sick_count * 0.21

    if random.random() < infection_rate:
        person.sick = True

#make move
def personMove(i, j):
    direction = random.random()
    if direction>=0 and direction<.20:
        if ((i-1)>0):
            if environment[i-1][j] == None:
                environment[i-1][j] = environment[i][j]
                environment[i][j] = None
    if direction>=.25 and direction<.50:
        if ((j+1)<len(environment[0])):
            if environment[i][j+1] == None:
                environment[i][j+1] = environment[i][j]
                environment[i][j] = None
    if direction>=.50 and direction<.75:
        if ((i+1)<len(environment)):
            if environment[i+1][j] == None:
                environment[i+1][j] = environment[i][j]
                environment[i][j] = None
    if direction>=.80 and direction<=1:
        if((j-1)>0):
            if environment[i][j-1] == None:
                environment[i][j-1] = environment[i][j]
                environment[i][j] = None

#When using get functions to retrieve surrounding cell information, this is the model it follows
#  0     1     2
#  3   TARGET  4
#  5     6     7
def get_zero(x, y):
    x-=1
    y-=1
    if (x<0) or (y<0):
        return None
    else:
        return environment[x][y]

def get_one(x, y):
    y-=1
    if (y<0):
        return None
    else:
        return environment[x][y]

def get_two(x, y):
    x+=1
    y-=1
    if (x>(len(environment)-1)) or (y<0):
        return None
    else:
        return environment[x][y]

def get_three(x, y):
    x-=1
    if (x<0):
        return None
    else:
        return environment[x][y]

def get_four(x, y):
    x+=1
    if (x>(len(environment)-1)):
        return None
    else:
        return environment[x][y]

def get_five(x, y):
    x-=1
    y+=1
    if (x<0) or (y>(len(environment)-1)):
        return None
    else:
        return environment[x][y]

def get_six(x, y):
    y+=1
    if (y>(len(environment)-1)):
        return None
    else:
        return environment[x][y]

def get_seven(x, y):
    x+=1
    y+=1
    if (x>(len(environment)-1)) or (y>(len(environment)-1)):
        return None
    else:
        return environment[x][y]


def get_surrounding(x,y):
    surround = [get_zero(x,y), get_one(x,y), get_two(x,y), get_three(x,y), get_four(x,y), get_five(x,y), get_six(x,y), get_seven(x,y)]
    return surround

def get_population():
    population = 0
    for i in range(len(environment)):
        for person in environment[i]:
            if person != None:
                population+=1
    return population

def get_healthy():
    healthy = 0
    for i in range(len(environment)):
        for person in environment[i]:
            if person != None:
                if not person.sick:
                    healthy+=1
    return healthy

def get_social_d_healthy():
    social_d_healthy = 0
    for i in range(len(environment)):
        for person in environment[i]:
            if person != None:
                if (not person.sick) and (person.social_distance):
                    social_d_healthy+=1
    return social_d_healthy

def get_not_social_d_healthy():
    not_social_d_healthy = 0
    for i in range(len(environment)):
        for person in environment[i]:
            if person != None:
                if (not person.sick) and (not person.social_distance):
                    not_social_d_healthy +=1
    return not_social_d_healthy

def get_infected():
    infected=0
    for i in range(len(environment)):
        for person in environment[i]:
            if person != None:
                if person.sick:
                    infected+=1
    return infected

def get_social_d_infected():
    vaccinated_infected = 0
    for i in range(len(environment)):
        for person in environment[i]:
            if person != None:
                if (person.sick) and (person.social_distance):
                    vaccinated_infected+=1
    return vaccinated_infected

def get_not_social_d_infected():
    unvaccinated_infected = 0
    for i in range(len(environment)):
        for person in environment[i]:
            if person != None:
                if (person.sick) and (not person.social_distance):
                    unvaccinated_infected+=1
    return unvaccinated_infected

def get_infection_deaths():
    return dead_from_sickness

def get_vaccinated_deaths():
    return social_dis_deaths

def get_unvaccinated_deaths():
    return not_social_dis_deaths

def get_recovered():
    return healed

initializeEnvironment()
writefile = open('result.csv', 'w+')
csvwriter = csv.writer(writefile)
fields = ["STEP", "POPULATION", "HEALTHY", "VACCINATED HEALTHY", "UNVACCINATED HEALTHY", "INFECTED", "VACCINATED INFECTED", "UNVACCINATED INFECTED", "DEATHS FROM INFECTION", "DEATHS VACCINATED", "DEATHS UNVACCINATED" ]
rows = []
step_count = 0

while step_count < 500:
    step()
    step_count += 1
    #get stats
    population = get_population()
    healthy = get_healthy()
    social_d_healthy = get_social_d_healthy()
    not_social_d_healthy = get_not_social_d_healthy()
    infected = get_infected()
    social_d_infected = get_social_d_infected()
    not_social_d_infected = get_not_social_d_infected()
    infected_deaths = get_infection_deaths()
    social_d_deaths = get_vaccinated_deaths()
    not_social_d_deaths = get_unvaccinated_deaths()
    recovered = get_recovered()

    print ("\n")
    print ("Step: " + str(step_count))
    print ("Population: " + str(population))
    print ("Healthy: " + str(healthy))
    print ("--Social Distancing: " + str(social_d_healthy))
    print ("--Not Social Distancing: " + str(not_social_d_healthy))
    print ("Infected: " + str(infected))
    print ("--Social Distancing: " + str(social_d_infected))
    print ("--Not Social Distancing: " + str(not_social_d_infected))
    print ("Deaths from infection: " + str(infected_deaths))
    print ("--Social Distancing: " + str(social_d_deaths))
    print ("--Not Social Distancing: " + str(not_social_d_deaths))
    print ("Recovered: " + str(recovered))

    row =[step_count, population, healthy, social_d_healthy, not_social_d_healthy, infected, social_d_infected, not_social_d_infected, infected_deaths, social_d_deaths, not_social_d_deaths]
    rows.append(row)
    if get_infected() == 0:
        break

    #time.sleep(.25)

csvwriter.writerow(fields)
csvwriter.writerows(rows)

