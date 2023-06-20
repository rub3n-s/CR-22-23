function trainFunction_a()
% Pasta do Dataset
DATASET_FOLDER = 'start';

% Numero de ficheiros de imagem por pasta
NUM_FILES = 5;

% Numero de pastas
NUM_FOLDERS = 14;

% Obter a matriz binaria e o matriz do target
[binaryMatrix,targetMatrix] = getBinaryMatrixTargetMatrix(DATASET_FOLDER,NUM_FOLDERS,NUM_FILES);

target = onehotencode(targetMatrix,1,'ClassNames',1:14);
in = binaryMatrix;

disp(target);

%% Treinar rede
% Testar com 10 camadas escondidas
net = feedforwardnet(10);

net.divideFcn = '';
net.trainFcn = 'traingdx';

%% Realizar 10 iteracoes de treino e calcular media
somaTreinos = 0;

for k=1:1
    fprintf('\n---------- Iteracao [%d] ----------\n',k);
    %% Treinar, Simular e Apresentar Resultados
    % Treinar 
    [net,tr] = train(net, in, target);    

    % Simular
    out = sim(net, in);
    
    r = 0;
    for i=1:size(out,2)
        [a b] = max(out(:,i));
        [c d] = max(target(:,i));
        if b == d
          r = r+1;
        end
    end
    
    accuracy = r/size(out,2) * 100;
    somaTreinos = somaTreinos + accuracy;
    fprintf('Precisao do Treino = %.2f\n', accuracy)

    plotconfusion(target,out) % Matriz de confusao

    %% Desempenho
    %disp(tr.performFcn);
    disp(['Train Performance: ' num2str(tr.best_perf)])
    disp(['Validation Performance: ' num2str(tr.best_vperf)]);
    disp(['Test Performance: ' num2str(tr.best_tperf)]);
end

%% Apresentar a Media
fprintf('\n---------- Apos 10 iteracoes ----------\n')
fprintf('Media de Precisao = %.2f\n',somaTreinos/10);

end