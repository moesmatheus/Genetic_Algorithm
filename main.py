import numpy as np
import time


#Function to generate random genes from options
def generate_genes(sample, size):

    randon_feature = lambda size, array: np.array(array)[np.random.randint(0, len(array), size)]

    return np.array(list(map(randon_feature, [size] * len(sample), sample))).transpose()


def cross(a, b):
    order = np.random.randint(low=0, high=2, size=len(a))
    cross = np.array(list(map(lambda a, b, n: a if n == 0 else b, a, b, order)))
    return cross


def pool(fit_gen, size_out=50):
    index = np.random.randint(low=0, high=len(fit_gen), size=[size_out, 2])
    new_genes = []
    for a, b in index:
        new_gene = cross(fit_gen[a], fit_gen[b])
        new_genes.append(new_gene)
    new_genes = np.array(new_genes)

    return new_genes

def mutation(genes,gene_options,p_mut = 0.1):

    mutation_genes = generate_genes(gene_options,len(genes))

    new_genes = []

    for a,b in zip(genes, mutation_genes):

        order = np.random.choice([0, 1], size=len(genes), p=[1 - p_mut, p_mut])

        new_gene = np.array(list(map(lambda a, b, n: a if n == 0 else b, a, b, order)))

        new_genes.append(new_gene)

    new_genes = np.array(new_genes)

    return new_genes




class GA():

    def __init__(self, pop_size, gene_options, fitness_function, polarity = 'Ascending', threshold = 10, mutation_rate = False):

        self.pop_size = pop_size
        self.polarity = polarity

        if self.polarity == 'Ascending':
            self.threshold = 100 - threshold
        else:
            self.threshold = threshold

        self.gene_options = gene_options
        self.mutation_rate = mutation_rate
        self.fitness_function = fitness_function
        self.genes = generate_genes(self.gene_options, self.pop_size)
        self.fitness_scores = np.array(list(map(self.fitness_function, self.genes)))
        self.generation_n = 0
        self.fittest_genes = []
        self.fit_seq = []
        self.fit_avg_seq = []


    def get_fitness(self):

        self.fitness_scores = np.array(list(map(self.fitness_function, self.genes)))

        return self.fitness_scores

    def do_generation(self,show = True):

        t = time.time()

        #Update generation number
        self.generation_n += 1

        #Get cut number for fitness score
        self.cut = np.percentile(self.fitness_scores,self.threshold)

        #Define wich are the fit genes
        if self.polarity == 'Ascending':

            fit_gen = np.array([g for g, f in zip(self.genes, self.fitness_scores) if f >= self.cut])

        else:

            fit_gen = np.array([g for g, f in zip(self.genes, self.fitness_scores) if f >= self.cut])

        #Create new generation of genes
        self.genes = pool(fit_gen,self.pop_size)

        #Make a mutation
        if self.mutation_rate != False:

            self.genes = mutation(self.genes, self.gene_options, self.mutation_rate)

        #Recalculate fitness scores
        self.fitness_scores = np.array(list(map(self.fitness_function, self.genes)))

        #Find the fittest gene
        self.fittest_gene = self.genes[
            np.argmax(self.fitness_scores)] if self.polarity == 'Ascending' else self.genes[np.argmin(self.fitness_scores)]
        
        self.fittest_genes.append(self.fittest_gene)

        #Find the best fitness score
        self.best_fit = np.max(self.fitness_scores) if self.polarity == 'Ascending' else np.min(self.fitness_scores)
        
        self.fit_seq.append(self.best_fit)
        self.fit_avg_seq.append(np.mean(self.fitness_scores))
        

        #Print generation numbers
        if show == True:
            print('#Generation: %i \t Best Fitness: %.4f \t Mean Fitness %.4f \t Execution time: %.4f'%
                  (self.generation_n,self.best_fit,np.mean(self.fitness_scores),time.time()-t))


    def batch_generation(self, n_do = 1, n_show = 1):

        for i in range(n_do):

            if i % n_show ==0:

                self.do_generation(show = True)

            else:

                self.do_generation(show = False)
