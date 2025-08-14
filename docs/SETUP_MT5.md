# Configurando o MetaTrader 5

1) **Instale o MT5** (MetaQuotes). Abra e **crie/entre** em uma **conta demo**.
2) **Habilite**:
   - Botão **Algo Trading** (verde) na barra superior.
   - `Ferramentas > Opções > Expert Advisors`:
     - [x] Permitir DLL imports
     - [x] Permitir WebRequest (se for usar APIs externas)
3) **Ativos no Market Watch**: clique direito > *Mostrar Tudo* para expor EURUSD, GBPUSD, USDJPY, etc.
4) **Mantenha o MT5 aberto** enquanto roda o bot.
5) **Permissões do Windows**: se usar antivírus restritivo, adicione exceção para o terminal MT5.
6) **Primeira conexão**: a lib `MetaTrader5` conecta automaticamente ao terminal aberto.  
   Se tiver múltiplas instalações, defina a variável de ambiente `MT5_TERMINAL_PATH` apontando para `terminal64.exe`.

> Dica: evite operar em horários de **rollover** (cerca de 21:00–22:15 UTC), spreads explodem e o bot pode filtrar tudo.