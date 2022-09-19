# SIwR projekt

Projekt polega na śledzącego przechodniów z wykorzystaniem probablistycznego modeleu grafowego.

Program odczytuje plik (bboxes.txt) z informacjami o poszczególnych klatkach nagrania. Następnie w pętli głównej interpretuje je klatka po klatce. Dane o danej w klatce zbierane są do słownika (dictionary). Zebrane informacje służą do zainicjowania klasy Frame. Powstała klasa zapisywana jest do listy, służącej do "zapamiętania" kilku ostatnich klatek (ostatecznie wykorzystywana jest tylko jedna, poprzednia klatka). Następnie uruchamiana jest metoda process_the_frame, która dalej przetwarza informacje. Kolejną czynnością w pętli jest uzyskanie wyników na podstawie obecnej i poprzedniej klatki za pomocą funkcji get_probability oraz wypisać jej na standardowe wyjście. W celach testowych powstał fragment programu wykonywany po ręcznym przypisaniu fladze testing_mode_flag wartości True. Wczytuje on pożądany wynik z osobnego dodatkowego pliku (bboxes_gt.txt). Wypisuje on w konsoli porównanie wyników oraz na koniec uzyskana dokładność.

Klasa Frame służy do przechowywania danych o danej klatce filmu oraz do pozyskiwania dodatkowych informacji o niej.
     
W funkcji get_probability porównywane są informacje o bounding box'ach na poprzedniej i obecnej klatce. Do nowo zainicjowanego grafu dodawane są węzły. Do węzłów przypisane są współczynniki z wektorami danych w postaci list, rozszerzone są one o stałą wartość 0.49 (dobrane eksperymentalnie) dla prawdopodobieństwa tego, że dany bounding box pojawił się pierwszy raz (probe_vec na rysunku 1). Powstałe współczynniki dodawanie są do grafu. Następnie za pomocą funkcji combinations pozyskiwane są wszystkie kombinacje węzłów (bez powtarzających się), na podstawie których włożone i dodawanie do grafu są krawędzie. Przy tworzeniu krawędzi stosowania jest macierz łącząca(link_mtx na rysunku 1). Za pomocą metody map_query z klasy BeliefPropagation na podstawie utworzonego grafu dokonywanie jest wnioskowanie. Uzyskane wyniki zmniejszanie są o jeden do dostosować je do żadnego formatu.

Czynniki jednoargumentowe (unary factors) służące do utworzenia wektora danych wejściowych (probe_vec na rysunku 1) są różnicą histogramu danego go bounding box'a z danym bounding box'em z poprzedniej klatki. Wartości są znormalizowaną (do przedziału 0 - 1). W celu zmniejszenia wpływu tła wycięte bounding box'y pomniejszane są o 15% (do około od krawędzi od środka) względem oryginału.

![graph](https://user-images.githubusercontent.com/50810180/191033742-b8f308f2-d61e-4c54-bf7a-45c17b4ada32.jpg) 

Obrazek 1. Przykładowy graf dla przypadku gdy obecna oraz poprzednia ramka ma po 3 bounding box'y.   
Gdzie n oznacza numer bounding box'a (przy czym wyrażenie n+1 dla n równego n_max oznacza 0)

Wyniki testów (kilka ostatnich linijek porównania oraz dokładność):
     est: [1, 2] gt: [1, -1]
     est: [0, 1] gt: [0, 1]
     est: [0, 1] gt: [0, 1]
     est: [0, 1] gt: [0, 1]
     est: [0, 1] gt: [0, 1]
     accuracy: 89.35%
