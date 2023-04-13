function user_interface()

%% Interface do Utilizador
% Pede digitos ao utilizador para chamar a funcao de treino
% Condicao de paragem
%   Digitar 'exit'

%% Limpar e Fechar
clc;
clear all;
close all;

%% Selecionar digito a ser treinado
while 1
    fprintf('\nSelecione qual digito pretende treinar ("exit" para sair): \n')
    fprintf('Opcoes = {0-9  +  -  /  *}\n')
    op = input('Digito: ','s');
    
    switch(op)
        % Opcao caso o utilizador deseje sair do ciclo
        case 'exit' 
            break
        % Caso seja uma opcao valida chama a funcao de treino
        case {'0','1','2','3','4','5','6','7','8','9'}
            train_function(op)
        case '+'
            train_function('add')
        case '-'
            train_function('sub')
        case '/'
            train_function('div')
        case '*'
            train_function('mul')
        % Se for uma opcao invalida pede denovo
        otherwise
            disp('Opcao invalida!')
    end
end
end