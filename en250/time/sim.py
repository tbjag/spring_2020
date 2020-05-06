# coding: utf-8
import random
import time
import csv
POPULATION_DENSITY = .75
VACCINATED_PERCENTAGE = .5
INITIAL_SICK_PERCENTAGE = 0.001

class Person:
    def __init__(self, alive, vaccinated, sick, age, sick_days):
        self.alive = alive
        self.vaccinated = vaccinated
        self.sick = sick
        self.age = age
        self.sick_days = sick_days


environment = [[None for _ in range(200)] for _ in range(100)]
dead_from_sickness = 0
healed = 0
vaccinated_deaths = 0
unvaccinated_deaths = 0
age_deaths = 0

def initializeEnvironment():
    global POPULATION_DENSITY
    global VACCINATED_PERCENTAGE
    global INITIAL_SICK_PERCENTAGE
    for i in range(len(environment)):
        for j in range(len(environment[i])):
            x = None
            exist_chance = random.random()
            vaccinated_chance = random.random()
            sick_chance = random.random()
            if random.random() < POPULATION_DENSITY: #percent chance there is a person in this location
                 x = Person(True, (vaccinated_chance < VACCINATED_PERCENTAGE), (sick_chance < INITIAL_SICK_PERCENTAGE), random.randint(1, 23725), 0)
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
                    break;

                #Log people that are sick and give them a small chance of healing
                determineHealth(person)
                surrounding_people = get_surrounding(i, j)
                determineInfectionRate(person, surrounding_people)
                personMove(i, j)

#The chance that someone will die depending on how many days they've been sick already. Each index is one day
death_rates = [0,.0125,.0156,.0195,.0244,.0305,.0381,.0476,.0596,.0745,.0931,.1164,.1455,.1818,.2273,.2842,.3552,.4440,.5551,.6938,.8673,1.0842]

#returns True if the person dies of old age
def determineDeath(person):
    global dead_from_sickness
    global vaccinated_deaths
    global unvaccinated_deaths
    global age_deaths
    if person != None:
        if person.age > 28470:
            age_deaths+=1
            return True
        if person.sick:
            if random.random() < death_rates[person.sick_days]:
                dead_from_sickness+=1
                if person.vaccinated:
                    vaccinated_deaths+=1
                else:
                    unvaccinated_deaths+=1
                return True
    return False


def determineHealth(person):
    global healed
    if person.sick:
        person.sick_days+=1
        #2.5 percent chance of healing everyday to add up to 50% over the 20 day perso
        if random.random() < .025:
            person.sick = False
            person.sick_days = 0
            healed+=1

#Changes the persons sick status based on the condition of the people around them
def determineInfectionRate(person, surrounding_people):
    sick_count = 0
    for neighbor in surrounding_people:
        if neighbor != None:
            if neighbor.sick:
                sick_count+=1

    if person.vaccinated:
        infection_rate = sick_count * 0.0125
    else:
        infection_rate = sick_count * 0.125

    if random.random() < infection_rate:
        person.sick = True

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


def print_pretty():
    for i in range(len(environment)):
        line = ""
        for person in environment[i]:
            if person != None:
                if person.sick:
                    line += u"\u2588" + ""
                else:
                    line += "-"
            else:
                line += " "
        print line

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

def get_vaccinated_healthy():
    vaccinated_healthy = 0
    for i in range(len(environment)):
        for person in environment[i]:
            if person != None:
                if (not person.sick) and (person.vaccinated):
                    vaccinated_healthy+=1
    return vaccinated_healthy

def get_unvaccinated_healthy():
    unvaccinated_healthy = 0
    for i in range(len(environment)):
        for person in environment[i]:
            if person != None:
                if (not person.sick) and (not person.vaccinated):
                    unvaccinated_healthy +=1
    return unvaccinated_healthy

def get_infected():
    infected=0
    for i in range(len(environment)):
        for person in environment[i]:
            if person != None:
                if person.sick:
                    infected+=1
    return infected

def get_vaccinated_infected():
    vaccinated_infected = 0
    for i in range(len(environment)):
        for person in environment[i]:
            if person != None:
                if (person.sick) and (person.vaccinated):
                    vaccinated_infected+=1
    return vaccinated_infected

def get_unvaccinated_infected():
    unvaccinated_infected = 0
    for i in range(len(environment)):
        for person in environment[i]:
            if person != None:
                if (person.sick) and (not person.vaccinated):
                    unvaccinated_infected+=1
    return unvaccinated_infected

def get_infection_deaths():
    return dead_from_sickness

def get_vaccinated_deaths():
    return vaccinated_deaths

def get_unvaccinated_deaths():
    return unvaccinated_deaths

def get_old_age_deaths():
    return age_deaths

def get_recovered():
    return healed

initializeEnvironment()
writefile = open('result.csv', 'w+')
csvwriter = csv.writer(writefile)
fields = ["STEP", "POPULATION", "HEALTHY", "VACCINATED HEALTHY", "UNVACCINATED HEALTHY", "INFECTED", "VACCINATED INFECTED", "UNVACCINATED INFECTED", "DEATHS FROM INFECTION", "DEATHS VACCINATED", "DEATHS UNVACCINATED", "DEATHS FROM OLD AGE"]
rows = []
step_count = 0

while True:
    step()
    step_count += 1
    print_pretty()

    population = get_population()
    healthy = get_healthy()
    vacc_healthy = get_vaccinated_healthy()
    unvacc_healthy = get_unvaccinated_healthy()
    infected = get_infected()
    vacc_infected = get_vaccinated_infected()
    unvacc_infected = get_unvaccinated_infected()
    infected_deaths = get_infection_deaths()
    vacc_deaths = get_vaccinated_deaths()
    unvacc_deaths = get_unvaccinated_deaths()
    old_age_deaths = get_old_age_deaths()
    recovered = get_recovered()

    print "\n"
    print "Step: " + str(step_count)
    print "Population: " + str(population)
    print "Healthy: " + str(healthy)
    print "--Vaccinated: " + str(vacc_healthy)
    print "--Unvaccinated: " + str(unvacc_healthy)
    print "Infected: " + str(infected)
    print "--Vaccinated: " + str(vacc_infected)
    print "--Unvaccinated: " + str(unvacc_infected)
    print "Deaths from infection: " + str(infected_deaths)
    print "--Vaccinated: " + str(vacc_deaths)
    print "--Unvaccinated: " + str(unvacc_deaths)
    print "Deaths from old age: " + str(old_age_deaths)
    print "Healed: " + str(recovered)
    #

    row =[step_count, population, healthy, vacc_healthy, unvacc_healthy, infected, vacc_infected, unvacc_infected, infected_deaths, vacc_deaths, unvacc_deaths, old_age_deaths]
    rows.append(row)
    if get_infected() == 0:
        break

    time.sleep(.25)

csvwriter.writerow(fields)
csvwriter.writerows(rows)

# print environment
# print "\n"
#
# total = 0
# for i in range(len(environment)):
#     total += environment[i].count(None)
#     print environment[i].count(None)
# print total
#
# print "AGE"
# for i in range(len(environment)):
#     for j in range(len(environment[i])):
#         try:
#             print environment[i][j].age / 365
#         except:
#             print "False"
