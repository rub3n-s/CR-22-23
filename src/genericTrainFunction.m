function [perfGlobal,perfTest] = genericTrainFunction(...
                                    camadasEscondidas, ...
                                    neuroniosCamada1, neuroniosCamada2, neuroniosCamada3, ...
                                    funcTreino,  funcAtivacao1, funcAtivacao2, funcAtivacao3, funcDivisao, ...
                                    trainRatio, val, test, ...
                                    pasta, nomeRede)
%% Informacao dos Parametros Recebidos
clc;
fprintf('Camadas Escondidas: %d\n',camadasEscondidas);
fprintf('Neuronios Camada 1: %d\n',neuroniosCamada1);
fprintf('Neuronios Camada 2: %d\n',neuroniosCamada2);
fprintf('Neuronios Camada 3: %d\n',neuroniosCamada3);
fprintf('Funcao de Treino: %s\n',funcTreino);
fprintf('Funcao de Ativacao 1: %s\n',funcAtivacao1);
fprintf('Funcao de Ativacao 2: %s\n',funcAtivacao2);
fprintf('Funcao de Ativacao 3: %s\n',funcAtivacao3);
fprintf('Funcao de Divisao: %s\n',funcDivisao);
fprintf('Pasta Dataset: %s\n',pasta);
fprintf('Nome Rede: %s\n',nomeRede);

%% Definir Constantes e Variaveis
% Resolucao das imagens
% Tamanho padrao das imagens 150x150
% Minimo 25x25
IMG_RES = [25 25];

% Numero de ficheiros de imagem por pasta
switch(pasta)
    case 'start'
        NUM_FILES = 5;
    case 'train'
        NUM_FILES = 50;
    case 'custom'
        NUM_FILES = 3;
end

% Numero de pastas
NUM_FOLDERS = 14;

% Gerar uma matriz
binaryMatrix = zeros(IMG_RES(1) * IMG_RES(2), NUM_FILES);
targetMatrix = [];
count = 1;

%% Ler, redimensionar e preparar os targets
fprintf('\nA ler imagens...\nPastas acessadas:\n');
for i=1:NUM_FOLDERS
    % Percorrer os ficheiros dentro da pasta i
    for j=1:NUM_FILES
        switch(i-1)
            case 10     % add
                file = fullfile('..','NN_datasets',DATASET_FOLDER,'add',sprintf('%d.png',j));
            case 11     % div
                file = fullfile('..','NN_datasets',DATASET_FOLDER,'div',sprintf('%d.png',j));
            case 12     % mul
                file = fullfile('..','NN_datasets',DATASET_FOLDER,'mul',sprintf('%d.png',j));
            case 13     % sub
                file = fullfile('..','NN_datasets',DATASET_FOLDER,'sub',sprintf('%d.png',j));
            otherwise   % [0,9]
                file = fullfile('..','NN_datasets',DATASET_FOLDER,sprintf('%d',i-1),sprintf('%d.png',j));
        end        
        img = imread(file);
        img = im2gray(img);
        img = imresize(img, IMG_RES);
        binarizedImg = imbinarize(img);
        binaryMatrix(:, count) = reshape(binarizedImg, 1, []);
        count=count+1;
    end
end

% Obter todos os vetores (cada um corresponde a uma pasta)
vec1 = repelem(1, NUM_FILES);       % 0
vec2 = repelem(2, NUM_FILES);       % 1
vec3 = repelem(3, NUM_FILES);       % 2
vec4 = repelem(4, NUM_FILES);       % 3
vec5 = repelem(5, NUM_FILES);       % 4
vec6 = repelem(6, NUM_FILES);       % 5
vec7 = repelem(7, NUM_FILES);       % 6
vec8 = repelem(8, NUM_FILES);       % 7
vec9 = repelem(9, NUM_FILES);       % 8
vec10 = repelem(10, NUM_FILES);     % 9
vec11 = repelem(11, NUM_FILES);     % +
vec12 = repelem(12, NUM_FILES);     % /
vec13 = repelem(13, NUM_FILES);     % *
vec14 = repelem(14, NUM_FILES);     % -

% Preencher a matriz a partir dos vetores
targetMatrix = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, ...
                vec8, vec9, vec10, vec11, vec12, vec13, vec14];

target = onehotencode(targetMatrix,1,'ClassNames',1:14);
in = binaryMatrix;

%% Treinar rede
% Testar com x neuronios e y camadas escondidas
switch(camadasEscondidas)
    case 1
        net = feedforwardnet(neuroniosCamada1);
    case 2
        net = feedforwardnet([neuroniosCamada1 neuroniosCamada2]);
    case 3
        net = feedforwardnet([neuroniosCamada1 neuroniosCamada2 neuroniosCamada3]);
end

%% Configurar a Rede
% Função de Ativação
net.layers{1}.transferFcn = funcAtivacao1;
net.layers{2}.transferFcn = funcAtivacao2;

if camadasEscondidas > 1
    net.layers{3}.transferFcn = funcAtivacao3;
end

% Numero de Epocas
net.trainParam.epochs = 100;

% Funcao de Treino
net.trainFcn = funcTreino;

% Divisao de Treino
net.divideFcn = funcDivisao;
net.divideParam.trainRatio = trainRatio;
net.divideParam.valRatio = val;
net.divideParam.testRatio = test;

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
    fprintf('\tPrecisao Global = %.2f\n', globalAccuracy)

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
end

%% Guardar a rede
net = netAux;
netFile = fullfile('..','networks',sprintf('net_%s_pmg%d_pmt%d',nomeRede,round(sumGlobal/10),round(sumTest/10)));
save(netFile, 'net');

%% Apresentar a Media
fprintf('\n---------- Apos 10 Iterações ----------\n')
fprintf('\tMedia Precisao Total = %.2f\n', sumGlobal/10);
fprintf('\tMedia Precisao Teste = %.2f\n', sumTest/10);
fprintf('\tMedia Performance Treino = %.2f\n', sumPerformanceTrain/10);
fprintf('\tMedia Performance Teste = %.2f\n', sumPerformanceTest/10);

perfGlobal = sumGlobal/10;
perfTest = sumTest/10;
end