package apresentacao;

import java.io.FileReader;
import java.util.Random;

import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;


public class ValidacaoCruzada {

	public static void main(String[] args) throws Exception {
		
		FileReader leitor = new FileReader("posicoes.arff");
		Instances jogadores = new Instances(leitor);
		
		jogadores.setClassIndex(14); 
		
		Instances jogadoresTreino = jogadores.trainCV(3, 0); 
		Instances jogadoresTeste = jogadores.testCV(3, 0); 
		
		IBk knn = new IBk(5);
		IB1 vizinho = new IB1();
		
		knn.buildClassifier(jogadoresTreino);
		vizinho.buildClassifier(jogadoresTreino);
		
		System.out.println("VERDADEIRO;KNN5;VIZINHO"); 
		for (int i = 0; i < jogadoresTeste.numInstances(); i++) {
			Instance teste = jogadoresTeste.instance(i); 
			System.out.print(teste.value(14) + ";"); 
			teste.setClassMissing();
			
			double knnValue = knn.classifyInstance(teste);
			double vizinhoValue = vizinho.classifyInstance(teste);
			
			System.out.println(knnValue + ";" + vizinhoValue);
		}
	}
	
}
