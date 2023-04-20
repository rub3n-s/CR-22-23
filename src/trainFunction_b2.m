function trainFunction_b2()
%% NOTA
%   Neste exemplo são utilizadas duas redes
%       Uma para os digitos de 0 a 9
%       Outra para os operadores
%% Definir Constantes e Variaveis
% Resolucao das imagens
% Tamanho padrao das imagens 150x150
% Minimo 25x25
IMG_RES = [25 25];

% Pasta do Dataset
%DATASET_FOLDER = 'start';
DATASET_FOLDER = 'train';
%DATASET_FOLDER = 'custom_draw';

% Caminho base do Dataset
BASE_PATH = sprintf('../NN_datasets/%s/',DATASET_FOLDER);

% Numero de ficheiros de imagem por pasta
switch(DATASET_FOLDER)
    case 'start'
        NUM_FILES = 5;
    case 'train'
        NUM_FILES = 50;
    case 'custom_draw'
        NUM_FILES = 3;
end

% Numero de pastas
NUM_DIGIT_FOLDERS = 10;
NUM_OPERATOR_FOLDERS = 4;

%% [Digitos] 
% Gerar uma matriz
binaryMatrix = zeros(IMG_RES(1) * IMG_RES(2), NUM_FILES);
targetMatrix = [];

% ======== Ler, redimensionar e preparar os targets ========
fprintf('\n[Digitos] A ler imagens...\nPastas acessadas:\n');
count = 1;
for i=1:NUM_DIGIT_FOLDERS
    % Apenas para debug (Mostrar o path da pasta que esta a ser acedida)
    disp(fullfile('..','NN_datasets',DATASET_FOLDER,sprintf('%d',i-1)));

    % Percorrer os ficheiros (50) dentro da pasta i
    for j=1:NUM_FILES
        % Definir o caminho para o ficheiro .png
        file = fullfile('..','NN_datasets',DATASET_FOLDER,sprintf('%d',i-1),sprintf('%d.png',j));
        img = imread(file);
        img = im2gray(img);
        img = imresize(img, IMG_RES);
        binarizedImg = imbinarize(img);
        binaryMatrix(:, count) = reshape(binarizedImg, 1, []);
        count=count+1;
    end
end

% Obter todos os vetores (cada um corresponde a uma pasta)
vec1 = repelem(1, NUM_FILES);
vec2 = repelem(2, NUM_FILES);
vec3 = repelem(3, NUM_FILES);
vec4 = repelem(4, NUM_FILES);
vec5 = repelem(5, NUM_FILES);
vec6 = repelem(6, NUM_FILES);
vec7 = repelem(7, NUM_FILES);
vec8 = repelem(8, NUM_FILES);
vec9 = repelem(9, NUM_FILES);
vec10 = repelem(10, NUM_FILES);

% Preencher a matriz a partir dos vetores
targetMatrix = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, ...
                vec8, vec9, vec10];

target = onehotencode(targetMatrix,1,'ClassNames',1:10);
in = binaryMatrix;

% ======== Treinar rede ========
% Testar com x neuronios e y camadas escondidas
netDigits = feedforwardnet([10]);

% ======== Configurar a Rede ========
% Função de Ativação
netDigits.layers{1}.transferFcn = 'tansig';
netDigits.layers{2}.transferFcn = 'purelin';
%netDigits.layers{3}.transferFcn = 'purelin';

% Numero de Epocas
netDigits.trainParam.epochs = 100;

% Funcao de Treino
netDigits.trainFcn = 'trainlm';

% Divisao de Treino
netDigits.divideFcn = 'dividerand';
netDigits.divideParam.trainRatio = 0.4;
netDigits.divideParam.valRatio = 0.3;
netDigits.divideParam.testRatio = 0.3;

% ======== Treinar, Simular e Apresentar Resultados ========
% Realizar 10 iteracoes de treino e calcular media
sumDigitsTest = 0;
sumDigitsGlobal = 0;

netDigitsGlobal = 0;
netDigitsTest = 0;

for k=1:10
    fprintf('\n----- [Digitos] Iteracao %d -----\n',k);
    % Treinar 
    [netDigits,tr] = train(netDigits, in, target);    

    % Simular
    out = sim(netDigits, in);
    
    r = 0;
    for i=1:size(out,2)
        [a, b] = max(out(:,i));
        [c, d] = max(target(:,i));
        if b == d
          r = r+1;
        end
    end
    
    globalAccuracy = r/size(out,2)*100;
    sumDigitsGlobal= sumDigitsGlobal + globalAccuracy;
    fprintf('\tPrecisao Global = %.2f\n', globalAccuracy);

    % SIMULAR A REDE APENAS NO CONJUNTO DE TESTE
    TInput = in(:, tr.testInd);
    TTargets = target(:, tr.testInd);
    
    out = sim(netDigits, TInput);
    
    % erro = perform(netDigits, out,TTargets);
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
    sumDigitsTest= sumDigitsTest + testAccuracy;
    fprintf('\tPrecisao Teste = %.2f\n',testAccuracy);

    %% Guardar Valores da Net
    % Guardar a primeira iteracao ou a net que tiver melhores valores de
    % precisao
    if (k == 1 || globalAccuracy > netDigitsGlobal && testAccuracy > netDigitsTest)
        netDigitsGlobal = globalAccuracy;
        netDigitsTest = testAccuracy;
        netDigitsAux = netDigits; 
    end    

    %plotconfusion(target,out) % Matriz de confusao
end

%% [Operadores] 
% Gerar uma matriz
binaryMatrix = zeros(IMG_RES(1) * IMG_RES(2), NUM_FILES);
targetMatrix = [];

% ======== Ler, redimensionar e preparar os targets ========
count = 1;
fprintf('\n[Operadores] A ler imagens...\nPastas acessadas:\n');
for i=1:NUM_OPERATOR_FOLDERS
    % Apenas para debug (Mostrar o path da pasta que esta a ser acedida)
    switch(i)
        case 1     % add
            disp(fullfile('..','NN_datasets',DATASET_FOLDER,'add'));
        case 2     % div
            disp(fullfile('..','NN_datasets',DATASET_FOLDER,'div'));
        case 3     % mul
            disp(fullfile('..','NN_datasets',DATASET_FOLDER,'mul'));
        case 4     % sub
            disp(fullfile('..','NN_datasets',DATASET_FOLDER,'sub'));
    end

    % Percorrer os 50 ficheiros dentro da pasta i
    for j=1:NUM_FILES
        % Definir o caminho para o ficheiro .png
        switch(i)
            case 1     % add
                file = fullfile('..','NN_datasets',DATASET_FOLDER,'add',sprintf('%d.png',j));
            case 2     % div
                file = fullfile('..','NN_datasets',DATASET_FOLDER,'div',sprintf('%d.png',j));
            case 3     % mul
                file = fullfile('..','NN_datasets',DATASET_FOLDER,'mul',sprintf('%d.png',j));
            case 4     % sub
                file = fullfile('..','NN_datasets',DATASET_FOLDER,'sub',sprintf('%d.png',j));
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
vec1 = repelem(1, NUM_FILES);
vec2 = repelem(2, NUM_FILES);
vec3 = repelem(3, NUM_FILES);
vec4 = repelem(4, NUM_FILES);

% Preencher a matriz a partir dos vetores
targetMatrix = [vec1, vec2, vec3, vec4];

target = onehotencode(targetMatrix,1,'ClassNames',1:4);
in = binaryMatrix;

% ======== Treinar rede ========
% Testar com x neuronios e y camadas escondidas
netOperators = feedforwardnet([10]);

% ======== Configurar a Rede ========
% Função de Ativação
netOperators.layers{1}.transferFcn = 'tansig';
netOperators.layers{2}.transferFcn = 'purelin';
%netOperators.layers{3}.transferFcn = 'purelin';

% Numero de Epocas
netOperators.trainParam.epochs = 100;

% Funcao de Treino
netOperators.trainFcn = 'trainlm';

% Divisao de Treino
netOperators.divideFcn = 'dividerand';
netOperators.divideParam.trainRatio = 0.4;
netOperators.divideParam.valRatio = 0.3;
netOperators.divideParam.testRatio = 0.3;

% ======== Treinar, Simular e Apresentar Resultados ========
% Realizar 10 iteracoes de treino e calcular media
sumOperatorsTest = 0;
sumOperatorsGlobal = 0;

netOperatorsGlobal = 0;
netOperatorsTest = 0;

for k=1:10
    fprintf('\n----- [Operadores] Iteracao %d -----\n',k);
    % Treinar 
    [netOperators,tr] = train(netOperators, in, target);    

    % Simular
    out = sim(netOperators, in);
    
    r = 0;
    for i=1:size(out,2)
        [a, b] = max(out(:,i));
        [c, d] = max(target(:,i));
        if b == d
          r = r+1;
        end
    end
    
    globalAccuracy = r/size(out,2)*100;
    sumOperatorsGlobal= sumOperatorsGlobal + globalAccuracy;
    fprintf('\tPrecisao Global = %.2f\n', globalAccuracy)

    % SIMULAR A REDE APENAS NO CONJUNTO DE TESTE
    TInput = in(:, tr.testInd);
    TTargets = target(:, tr.testInd);
    
    out = sim(netDigits, TInput);
    
    % erro = perform(netDigits, out,TTargets);
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
    sumOperatorsTest= sumOperatorsTest + testAccuracy;
    fprintf('\tPrecisao Teste = %.2f\n', testAccuracy); 

    %% Guardar Valores da Net
    % Guardar a primeira iteracao ou a net que tiver melhores valores de
    % precisao
    if (k == 1 || globalAccuracy > netOperatorsGlobal && testAccuracy > netOperatorsTest)
        netOperatorsGlobal = globalAccuracy;
        netOperatorsTest = testAccuracy;
        netOperatorsAux = netOperators; 
    end    

    %plotconfusion(target,out) % Matriz de confusao
end

%% [Digitos] Apresentar a Media
fprintf('\n ------ [Digitos] Apos 10 iteracoes ------\n');
fprintf('\tMedia de Precisao Global = %.2f\n', sumDigitsGlobal/10);
fprintf('\tMedia de Precisao Teste = %.2f\n', sumDigitsTest/10);

% Guardar a rede
net = netDigitsAux;
netFile = fullfile('..','networks',sprintf('netDigits_%s_pmg%d_pmt%d',DATASET_FOLDER,round(sumDigitsGlobal/10),round(sumDigitsTest/10)));
save(netFile, 'net');

%% [Operadores] Apresentar a Media
fprintf('\n ------ [Operadores] Apos 10 iteracoes ------\n');
fprintf('\tMedia de Precisao Global = %.2f\n', sumOperatorsGlobal/10);
fprintf('\tMedia de Precisao Teste = %.2f\n', sumOperatorsTest/10);

% Guardar a rede
net = netOperatorsAux;
netFile = fullfile('..','networks',sprintf('netOperators_%s_pmg%d_pmt%d',DATASET_FOLDER,round(sumOperatorsGlobal/10),round(sumOperatorsTest/10)));
save(netFile, 'net');

end