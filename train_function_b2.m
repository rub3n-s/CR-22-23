function train_function_b2()
%% NOTA
%   Neste exemplo são utilizadas duas redes
%       Uma para os digitos de 0 a 9
%       Outra para os operadores
%% Definir Constantes e Variaveis
% Resolucao das imagens
% Tamanho padrao das imagens 150x150
% Minimo 25x25
IMG_RES = [25 25];

% Numero de ficheiros de imagem por pasta
NUM_FILES = 50;

% Numero de pastas
NUM_DIGIT_FOLDERS = 10;
NUM_OPERATOR_FOLDERS = 4;

%% [Digitos] 
% Gerar uma matriz
binaryMatrix = zeros(IMG_RES(1) * IMG_RES(2), NUM_FILES);
targetMatrix = [];

% ======== Ler, redimensionar e preparar os targets ========
count = 1;
for i=1:NUM_DIGIT_FOLDERS
    % Definir o caminho para a pasta
    FOLDER_PATH = sprintf('NN_datasets/train/%d/',i-1);    

    % Percorrer os ficheiros (50) dentro da pasta i
    for j=1:NUM_FILES
        img = imread(strcat(FOLDER_PATH,sprintf('%d.png',j)));
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
net = feedforwardnet([5 5]);

% ======== Configurar a Rede ========
% Função de Ativação
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';
net.layers{3}.transferFcn = 'purelin';

% Numero de Epocas
net.trainParam.epochs = 100;

% Funcao de Treino
net.trainFcn = 'trainlm';

% Divisao de Treino
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

% ======== Treinar, Simular e Apresentar Resultados ========
% Realizar 10 iteracoes de treino e calcular media
sumDigitsTest = 0;
sumDigitsGlobal = 0;
for k=1:10
    fprintf('\n----- [Digitos] Iteracao %d -----\n',k);
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
    
    accuracy = r/size(out,2)*100;
    sumDigitsGlobal= sumDigitsGlobal + accuracy;
    fprintf('\tPrecisao Global = %.2f\n', accuracy);

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

    accuracy = r/size(tr.testInd,2)*100;
    sumDigitsTest= sumDigitsTest + accuracy;
    fprintf('\tPrecisao Teste = %.2f\n',accuracy);

    %plotconfusion(target,out) % Matriz de confusao
end

%% [Operadores] 
% Gerar uma matriz
binaryMatrix = zeros(IMG_RES(1) * IMG_RES(2), NUM_FILES);
targetMatrix = [];

% ======== Ler, redimensionar e preparar os targets ========
count = 1;
for i=1:NUM_OPERATOR_FOLDERS
    % Definir o caminho para a pasta
    switch(i)
        case 1 % add
            FOLDER_PATH = 'NN_datasets/train/add/';
        case 2 % div
            FOLDER_PATH = 'NN_datasets/train/div/';
        case 3 % mul
            FOLDER_PATH = 'NN_datasets/train/mul/';
        case 4 % sub
            FOLDER_PATH = 'NN_datasets/train/sub/';
    end

    % Percorrer os 50 ficheiros dentro da pasta i
    for j=1:NUM_FILES
        img = imread(strcat(FOLDER_PATH,sprintf('%d.png',j)));
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
netOperators = feedforwardnet([5 5]);

% ======== Configurar a Rede ========
% Função de Ativação
netOperators.layers{1}.transferFcn = 'tansig';
netOperators.layers{2}.transferFcn = 'purelin';
netOperators.layers{3}.transferFcn = 'purelin';

% Numero de Epocas
netOperators.trainParam.epochs = 100;

% Funcao de Treino
netOperators.trainFcn = 'trainlm';

% Divisao de Treino
netOperators.divideFcn = 'dividerand';
netOperators.divideParam.trainRatio = 0.70;
netOperators.divideParam.valRatio = 0.15;
netOperators.divideParam.testRatio = 0.15;

% ======== Treinar, Simular e Apresentar Resultados ========
% Realizar 10 iteracoes de treino e calcular media
sumOperatorsTest = 0;
sumOperatorsGlobal = 0;
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
    
    accuracy = r/size(out,2)*100;
    sumOperatorsGlobal= sumOperatorsGlobal + accuracy;
    fprintf('\tPrecisao Global = %.2f\n', accuracy)

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

    accuracy = r/size(tr.testInd,2)*100;
    sumOperatorsTest= sumOperatorsTest + accuracy;
    fprintf('\tPrecisao Teste = %.2f\n', accuracy); 

    %plotconfusion(target,out) % Matriz de confusao
end

%% [Digitos] Apresentar a Media
fprintf('\n ------ [Digitos] Apos 10 iteracoes ------\n');
fprintf('\tMedia de Precisao Teste = %.2f\n', sumDigitsTest/10);
fprintf('\tMedia de Precisao Global = %.2f\n', sumDigitsGlobal/10);

%% [Operadores] Apresentar a Media
fprintf('\n ------ [Operadores] Apos 10 iteracoes ------\n');
fprintf('\tMedia de Precisao Teste = %.2f\n', sumOperatorsTest/10);
fprintf('\tMedia de Precisao Global = %.2f\n', sumOperatorsGlobal/10);

end