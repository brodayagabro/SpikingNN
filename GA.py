import numpy as np

class Genitor:
    # Genitor algorythm
    # Решение задачи одномерной полиноминалной регрессии
    loss_history : np.ndarray
    
    def __init__( self,
        population_size : int,
        chromosome_size : int,
        generations : int,
        mutation_rate : float,
        crossover_d : float,
        adaptive = True
    ):
        self.population_size = population_size;
        self.population_idices = np.arange(population_size)
        self.chromosome_size = chromosome_size;

        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_d = crossover_d 
        self.best_chromosome = np.empty(chromosome_size)
        
        # init population
        self.population = np.random.uniform(-5, 5, (population_size, chromosome_size))
        self.loss_history = np.zeros(generations)
        self.loss = np.zeros(population_size)

    def MSE(self, x : np.ndarray, y : np.ndarray):
        sqr = -(x - y)**2
        return np.mean(sqr)

    def fitness(self, chromosome, X, Y):
        x = np.polyval(chromosome, X)
        y = Y
        return self.MSE(x, y)
        
    def predict(self, X):
        return np.polyval(self.best_chromosome, X)

    def select_parents(self, fitnesses):
        parents = np.random.choice(self.population_idices, size=2, replace=False)
        return parents[0], parents[1]

    def crossover(self, p1, p2):
        d = self.crossover_d
        alpha = np.random.uniform(-d, 1+d, self.chromosome_size)
        child = alpha*p1 + (1-alpha)*p2
        return child

    def mutate(self, child, mutation_rate, mutation_step):
        n = 2*np.random.randint(0, 2, self.chromosome_size)-1 # signature of mutation
        p = np.random.random(self.chromosome_size) # probability of mutation
        child1 = np.where(p >= mutation_rate, child+n*mutation_step, child)
        return child1
        
    def fit(self, X_train, Y_train, mutation_rate=0.01, mutation_step=0.5):
        
        new_population = np.zeros((self.population_size+1, self.chromosome_size))
        new_population[:-1] = self.population
        fitnesses = np.empty(self.population_size+1)
        for generation in range(self.generations):
            # calc
            fitnesses[:-1] = np.array([self.fitness(chr, X_train, Y_train) for chr in self.population])
            self.loss_history[generation] = np.mean(fitnesses[:-1])
            # select random parents ids
            pid1, pid2 = self.select_parents(fitnesses)
            child = self.crossover(self.population[pid1], self.population[pid2])
            # mutate child
            child = self.mutate(child, mutation_rate, mutation_step)
            
            # select the less fit individ from population
            worse_id = np.argmin(fitnesses)
            new_population[-1] = child
            fitnesses[-1] = self.fitness(child, X_train, Y_train)
            new_population[:] = new_population[np.argsort(fitnesses)[::-1]]
            self.population[:] = new_population[:-1]

            fitnesses[:-1] = np.array([self.fitness(chr, X_train, Y_train) for chr in self.population])
            best_id = np.argmax(fitnesses[:-1]) 
            self.best_chromosome[:] =  self.population[best_id]

        return None

class Genitor_PID:
    # Genitor algorythm
    # Оптимизция системы с пидрегулятором и отрицательной обратой связью
    loss_history : np.ndarray
    
    def __init__( self,
        population_size : int,
        generations : int,
        crossover_d : float,
        adaptive = True
    ):
        self.population_size = population_size;
        self.population_idices = np.arange(population_size)
        self.chromosome_size = 3;

        self.generations = generations
        self.crossover_d = crossover_d 
        self.best_chromosome = np.empty(3)
        
        # init population
        self.population = np.random.uniform(0, 1, (population_size, 3))
        self.loss_history = np.zeros(generations)
        self.adaptive=adaptive

    def MSE(self, x : np.ndarray, y : np.ndarray):
        sqr = -(x - y)**2 # ???
        return np.mean(sqr)

    def fitness(self, chromosome, sys_params):
        
        PID_num = chromosome # PID - coefs
        PID_den = [1, 0]
        num = sys_params['num']
        den = sys_params['den']
        interm_num = np.convolve(num, PID_num)
        interm_den = np.convolve(den, PID_den)
        total_sys = signal.tf2ss(interm_den, np.polyadd(interm_den, interm_num))
        
        dT = 0.1
        # Моделирование системы происходит во времени 10 секунд
        # На основании этого строиться средний квартрат отклонения от нуля
        TotalTime = 10
        Tin = np.linspace(0, TotalTime, int(TotalTime/dT + 1))
        zero_input = np.zeros(Tin.shape) # не будем прилагать момент
        start_pos = 1 / 180.0*np.pi # начнём с позиции в 1°.
        Tout,yout,xout = signal.lsim(total_sys, zero_input, Tin, X0=[0, start_pos/40, 0])
        return self.MSE(yout, 0)

    def select_parents(self, fitnesses):
        parents = np.random.choice(self.population_idices, size=2, replace=False)
        return parents[0], parents[1]

    def crossover(self, p1, p2):
        d = self.crossover_d
        alpha = np.random.uniform(-d, 1+d, self.chromosome_size)
        child = alpha*p1 + (1-alpha)*p2
        return child

    def mutate(self, child, mutation_rate, mutation_step):
        n = 2*np.random.randint(0, 2, self.chromosome_size)-1 # signature of mutation
        p = np.random.random(self.chromosome_size) # probability of mutation
        child1 = np.where(p >= mutation_rate, child+n*mutation_step, child)
        return child1

    # На вход обучающей функции подается числитель и знаменатель передаточной функции системы
    def fit(self, num, den, mutation_rate=0.01, mutation_step=0.5, tau=10):
        
        new_population = np.zeros((self.population_size+1, self.chromosome_size))
        new_population[:-1] = self.population
        fitnesses = np.empty(self.population_size+1)
        sys_params = {
            'num': num,
            'den': den
        }
        mutation_st = mutation_step
        for generation in range(self.generations):
            if self.adaptive:
                mutation_st *= np.exp(-generation/tau)
            # calc
            fitnesses[:-1] = np.array([self.fitness(chr, sys_params) for chr in self.population])
            self.loss_history[generation] = np.mean(fitnesses[:-1])
            # select random parents ids
            pid1, pid2 = self.select_parents(fitnesses)
            child = self.crossover(self.population[pid1], self.population[pid2])
            # mutate child
            child = self.mutate(child, mutation_rate, mutation_st)
            
            # select the less fit individ from population
            worse_id = np.argmin(fitnesses)
            new_population[-1] = child
            fitnesses[-1] = self.fitness(child, sys_params)
            new_population[:] = new_population[np.argsort(fitnesses)[::-1]]
            self.population[:] = new_population[:-1]

            fitnesses[:-1] = np.array([self.fitness(chr, sys_params) for chr in self.population])
            best_id = np.argmax(fitnesses[:-1]) 
            self.best_chromosome[:] =  self.population[best_id]
            
        return self.best_chromosome