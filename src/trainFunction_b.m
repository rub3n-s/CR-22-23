function trainFunction_b()
% Pasta do Dataset
%DATASET_FOLDER = 'start';
%DATASET_FOLDER = 'train';
DATASET_FOLDER = 'train1';
%DATASET_FOLDER = 'custom_draw';

% Numero de ficheiros de imagem por pasta
switch(DATASET_FOLDER)
    case 'start'
        NUM_FILES = 5;
    case 'train'
        NUM_FILES = 50;
    case 'train1'
        NUM_FILES = 50;
    case 'custom_draw'
        NUM_FILES = 3;
end

% Numero de pastas
NUM_FOLDERS = 14;

% Obter a matriz binaria e o matriz do target
[binaryMatrix,targetMatrix] = getBinaryMatrixTargetMatrix(DATASET_FOLDER,NUM_FOLDERS,NUM_FILES);

target = onehotencode(targetMatrix,1,'ClassNames',1:14);
in = binaryMatrix;

%% Treinar rede
% Testar com x neuronios e y camadas escondidas
net = feedforwardnet([10 10 10]);

%% Configurar a Rede
% Função de Ativação
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'purelin';

% Funções de Ativacao:
%   tansig
%   purelin;
%   logsig;
%   hardlim;
%   hardlims;
%   compet;
%   elliotsig;

% Numero de Epocas
net.trainParam.epochs = 100;

% Funcao de Treino
net.trainFcn = 'trainlm';
%net.trainFcn = 'trainbfg';
%net.trainFcn = 'traingd';
%net.trainFcn = 'trainscg';
%net.trainFcn = 'trainoss';

% Divisao de Treino
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

%% Realizar 10 iteracoes de treino e calcular media
sumGlobal = 0;
sumTest = 0;
sumPerformanceTest = 0;
sumPerformanceTrain = 0;

netGlobal = 0;
netTest = 0;

for k=1:10
    fprintf('\n---------- Iteracao [%d] ----------\n',k);    
    %% Treinar, Simular e Apresentar Resultados
    % Treinar 
    [net,tr] = train(net, in, target);    

    % Simular
    out = sim(net, in);
    
    r = 0;
    for i=1:size(out,2)
        [a, b] = max(out(:,i));
        [c, d] = max(target(:,i));
        if b == d
          r = r+1;
        end
    end
    
    globalAccuracy = r/size(out,2)*100;
    sumGlobal= sumGlobal + globalAccuracy;
    fprintf('\tPrecisao Global = %.2f\n', globalAccuracy);

    %plotconfusion(target,out) % Matriz de confusao

    % SIMULAR A REDE APENAS NO CONJUNTO DE TESTE
    TInput = in(:, tr.testInd);
    TTargets = target(:, tr.testInd);
    
    out = sim(net, TInput);
    
    % erro = perform(net, out,TTargets);
    % fprintf('Erro na classificação do conjunto de teste %f\n', erro)
    
    %Calcula e mostra a percentagem de classificacoes corretas no conjunto de teste
    r=0;
    for i=1:size(tr.testInd,2)        % Para cada classificacao  
      [a b] = max(out(:,i));          % b guarda a linha onde encontrou valor mais alto da saida obtida
      [c d] = max(TTargets(:,i));     % d guarda a linha onde encontrou valor mais alto da saida desejada
      if b == d                       % se estao na mesma linha, a classificacao foi correta (incrementa 1)
          r = r+1;
      end
    end

    testAccuracy = r/size(tr.testInd,2)*100;
    sumTest = sumTest + testAccuracy;
    fprintf('\tPrecisao Teste = %.2f\n', testAccuracy);    

    %% Desempenho
    %disp(tr.performFcn);
    disp(['Train Performance: ' num2str(tr.best_perf)])
    disp(['Validation Performance: ' num2str(tr.best_vperf)]);
    disp(['Test Performance: ' num2str(tr.best_tperf)]);
    
    % Soma das performances para obter a media no final das 10 iteracoes
    sumPerformanceTrain = sumPerformanceTrain + tr.best_perf;
    sumPerformanceTest = sumPerformanceTest + tr.best_tperf;

    %% Guardar Valores da Net
    % Guardar a primeira iteracao ou a net que tiver melhores valores de
    % precisao
    if (k == 1 || globalAccuracy > netGlobal && testAccuracy > netTest)
        netGlobal = globalAccuracy;
        netTest = testAccuracy;
        netAux = net; 
    end

    %plotconfusion(target,out) % Matriz de confusao
end

%% Guardar a rede
net = netAux;
netFile = fullfile('..','networks',sprintf('netB_%s_pmg%d_pmt%d',DATASET_FOLDER,round(sumGlobal/10),round(sumTest/10)));
save(netFile, 'net');

%% Apresentar a Media
fprintf('\n---------- Apos 10 Iterações ----------\n')
fprintf('\tMedia Precisao Total = %.2f\n', sumGlobal/10);
fprintf('\tMedia Precisao Teste = %.2f\n', sumTest/10);
fprintf('\tMedia Performance Treino = %.2f\n', sumPerformanceTrain/10);
fprintf('\tMedia Performance Teste = %.2f\n', sumPerformanceTest/10);
end