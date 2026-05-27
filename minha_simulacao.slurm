#!/bin/bash 

#====================================================================                               
# 1. DIRETIVAS SBATCH (Instruções para o SLURM)      
#==================================================================== 

# Define o nome do Job                                                                                                                             
SBATCH --job-name=meu_teste 

# Redireciona a saída padrão (stdout) para um arquivo                                                                
SBATCH --output=resultado_%j.out 

# Redireciona a saída de erro (stderr) para um arquivo                                                             
SBATCH --error=erros_%j.err 

# Solicita o número de nós computacionais                                                                                  
SBATCH --nodes=1 

# Solicita o número de tarefas (processos) por nó                                                                             
SBATCH --ntasks-per-node=16  
                                                                                                                            
# Especifica a Qualidade de Serviço (QOS). PARÂMETRO OBRIGATÓRIO.                                                  
# Deve ser utilizado o QOS instruído pelo administrador.                                                      
 SBATCH --qos=normal 

# Define o tempo máximo de execução (formato: DD-HH:MM:SS)                                                
 SBATCH --time=00-00:10:00 

#====================================================================                                      
# 2. COMANDOS A SEREM EXECUTADOS PELO JOB                    
#==================================================================== 

echo "==============================================================="                                 
echo "Job iniciado em: $(date)" echo "Executando no nó: $(hostname)"                                   
echo "Diretório de submissão: $(pwd)"                                                                                                            
echo "===============================================================" 

# Assegura um ambiente limpo e carrega os módulos necessários module purge module load gnu14/14.2.0 

# Comando de execução do programa compilado                                                                             
 ./meuExecutavel 

echo "==============================================================="                          
echo "Job finalizado em: $(date)"                                                                                                                    
echo "==============================================================="  