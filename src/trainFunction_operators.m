function output = trainFunction_d(imgPath,netPath) 

disp(['A iniciar leitura do ficheiro ', imgPath]);

% Carregar a rede previamente treinada
load(netPath, 'net');

% Obtem o input
in = binarizedImageInput(imgPath, 0);

% Simular
out = sim(net, in);

% b - index de cada array
% Cada array corresponde a um digito
[a, b] = max(out);

switch(b)
    case 1
        disp('Resultado:  +');
        output = '+';
    case 2
        disp('Resultado:  /');
        output = '/';
   case 3
        disp('Resultado:  *');
        output = '*';
   case 4
        disp('Resultado:  -');
        output = '-';
    otherwise
        disp('Nao foi possivel reconhecer');
        output = 'Nao foi possivel reconhecer';
end
end