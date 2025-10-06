import pygad
import numpy
import time

# ======================================================
# 1. DEFINIÇÃO E GERAÇÃO DO PROBLEMA
# ======================================================

# Parâmetros para geração do problema
numero_de_itens = 40
capacidade_mochila = 100.0

# Define uma semente para garantir a reprodutibilidade dos resultados
numpy.random.seed(42)

# Gera os vetores de pesos e valores para cada item
pesos_itens = numpy.round(numpy.random.uniform(0.5, 15.0, numero_de_itens), 2)
valores_itens = numpy.random.randint(10, 150, numero_de_itens)

# Cria uma lista de dicionários para facilitar a exibição do resultado final
itens = [{'item': f'Item-{i+1}', 'peso': pesos_itens[i], 'valor': valores_itens[i]}
         for i in range(numero_de_itens)]

# ======================================================
# 2. FUNÇÕES DO ALGORITMO GENÉTICO
# ======================================================

def fitness_func(ga_instance, solution, solution_idx):
    """
    Calcula o fitness (aptidão) de uma solução. O fitness é o valor total
    dos itens na mochila. Soluções que excedem a capacidade são penalizadas
    com fitness zero, tornando-as inviáveis.
    """
    peso_total = numpy.sum(solution * pesos_itens)
    valor_total = numpy.sum(solution * valores_itens)

    if peso_total > capacidade_mochila:
        return 0
    else:
        return valor_total

def on_gen_callback(ga_instance):
    """
    Callback executado ao final de cada geração para reportar o progresso.
    Exibe o melhor fitness da geração atual a cada 10 gerações.
    """
    generation = ga_instance.generations_completed
    if generation % 10 == 0:
        best_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
        print(f"Geração {generation:3}: Melhor Fitness = {best_fitness:.2f}")

# ======================================================
# 3. CONFIGURAÇÃO E EXECUÇÃO DO AG
# ======================================================

# Parâmetros do Algoritmo Genético
ga_instance = pygad.GA(
    num_generations=200,
    num_parents_mating=20,
    sol_per_pop=100,
    num_genes=numero_de_itens,
    fitness_func=fitness_func,
    on_generation=on_gen_callback,
    gene_space=[0, 1],
    gene_type=int,
    parent_selection_type="sss",
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=8
)

# Inicia a execução do algoritmo e mede o tempo
print("Iniciando a execução do Algoritmo Genético...")
start_time = time.time()
ga_instance.run()
execution_time = time.time() - start_time
print("Execução concluída.")

# ======================================================
# 4. EXIBIÇÃO DOS RESULTADOS
# ======================================================

solution, solution_fitness, solution_idx = ga_instance.best_solution()
peso_selecionado = numpy.sum(solution * pesos_itens)

print("\n------------------- RESULTADO FINAL -------------------")
print(f"Tempo de Execução: {execution_time:.4f} segundos")
print(f"Melhor fitness (Valor Total): {solution_fitness:.2f}")
print(f"Peso Total da Solução: {peso_selecionado:.2f} kg (Capacidade Máxima: {capacidade_mochila} kg)")
print(f"Número de Itens na Mochila: {int(numpy.sum(solution))}")
print("-------------------------------------------------------")

print("\nItens selecionados:")
for i in range(len(solution)):
    if solution[i] == 1:
        print(f"  - {itens[i]['item']} (Peso: {itens[i]['peso']}, Valor: {itens[i]['valor']})")

# Gera o gráfico da evolução do fitness ao longo das gerações
ga_instance.plot_fitness(save_dir="fitness_plot.png")