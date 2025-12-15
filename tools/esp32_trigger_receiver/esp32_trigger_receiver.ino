 /*
 * ESP32 — TRIGGERS via Serial
 * LEFT  -> GPIO19  (ativa por 3 segundos)
 * RIGHT -> GPIO22  (ativa por 3 segundos)
 * 
 * Sistema de bloqueio: Após receber um comando, bloqueia por 3 segundos
 * e ignora qualquer outro sinal nesse período
 */

#define TRIGGER_LEFT_PIN   19   // GPIO19 para LEFT
#define TRIGGER_RIGHT_PIN  22   // GPIO22 para RIGHT
#define LED_PIN             2   // LED onboard para feedback

#define BAUD_RATE          115200
#define TRIGGER_DURATION   3000  // 3 segundos em ms (mudou de 100ms para 3000ms)

// Variáveis de controle de bloqueio
unsigned long last_trigger_time = 0;
bool trigger_active = false;

void setup() {
  Serial.begin(BAUD_RATE);

  pinMode(TRIGGER_LEFT_PIN, OUTPUT);
  pinMode(TRIGGER_RIGHT_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);

  digitalWrite(TRIGGER_LEFT_PIN, LOW);
  digitalWrite(TRIGGER_RIGHT_PIN, LOW);
  digitalWrite(LED_PIN, LOW);

  Serial.println("ESP32 BCI Trigger System v2.0");
  Serial.println("==============================");
  Serial.println("Modo: TRIGGER COM BLOQUEIO DE 3 SEGUNDOS");
  Serial.println("Comandos:");
  Serial.println("- TRIGGER_LEFT  | LEFT  | L");
  Serial.println("- TRIGGER_RIGHT | RIGHT | R");
  Serial.println("- PING");
  Serial.println("---");
  Serial.println("Ao receber comando:");
  Serial.println("  1. Ativa pino por 3 segundos");
  Serial.println("  2. Bloqueia novo sinal por 3 segundos");
  Serial.println("  3. Ignora comandos durante bloqueio");
  Serial.println("Use newline (\\n) no final do comando.");
  blinkBoot();
}

void loop() {
  static String buf;

  // Verificar se trigger expirou
  if (trigger_active && (millis() - last_trigger_time >= TRIGGER_DURATION)) {
    // Desativar pinos
    digitalWrite(TRIGGER_LEFT_PIN, LOW);
    digitalWrite(TRIGGER_RIGHT_PIN, LOW);
    digitalWrite(LED_PIN, LOW);
    trigger_active = false;
    Serial.println("[TRIGGER LIBERADO] Pode receber novo sinal");
  }

  // Ler dados seriais
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      buf.trim();
      if (buf.length() > 0) {
        processCommand(buf);
      }
      buf = "";
    } else {
      buf += c;
    }
  }
}

void processCommand(String command) {
  command.trim();
  command.toUpperCase();

  Serial.print("[RX] Comando recebido: ");
  Serial.println(command);

  // Verificar se está bloqueado
  if (trigger_active) {
    unsigned long tempo_restante = TRIGGER_DURATION - (millis() - last_trigger_time);
    Serial.print("[BLOQUEADO] Aguarde ");
    Serial.print(tempo_restante);
    Serial.println("ms para enviar novo sinal");
    return;  // Ignora comando
  }

  // Processar comando
  if (command == "TRIGGER_LEFT" || command == "LEFT" || command == "L") {
    executeTriggerLeft();
  }
  else if (command == "TRIGGER_RIGHT" || command == "RIGHT" || command == "R") {
    executeTriggerRight();
  }
  else if (command == "PING") {
    executePing();
  }
  else {
    Serial.print("[ERRO] Comando desconhecido - ");
    Serial.println(command);
  }
}

void executeTriggerLeft() {
  Serial.println("[EXEC] Executando TRIGGER_LEFT (GPIO19)");
  
  // Ativar pinos
  digitalWrite(TRIGGER_LEFT_PIN, HIGH);
  digitalWrite(LED_PIN, HIGH);
  
  // Registrar tempo e marcar como ativo
  last_trigger_time = millis();
  trigger_active = true;
  
  Serial.print("[BLOQUEIO] Sinal ativo por ");
  Serial.print(TRIGGER_DURATION);
  Serial.println("ms - Sistema bloqueado");
}

void executeTriggerRight() {
  Serial.println("[EXEC] Executando TRIGGER_RIGHT (GPIO22)");
  
  // Ativar pinos
  digitalWrite(TRIGGER_RIGHT_PIN, HIGH);
  digitalWrite(LED_PIN, HIGH);
  
  // Registrar tempo e marcar como ativo
  last_trigger_time = millis();
  trigger_active = true;
  
  Serial.print("[BLOQUEIO] Sinal ativo por ");
  Serial.print(TRIGGER_DURATION);
  Serial.println("ms - Sistema bloqueado");
}

void executePing() {
  Serial.println("[PING] PONG - ESP32 ativo e funcionando");
  
  // PING não ativa bloqueio - pode ser usado para testar conexão
  if (trigger_active) {
    unsigned long tempo_restante = TRIGGER_DURATION - (millis() - last_trigger_time);
    Serial.print("[STATUS] Trigger ativo - bloqueado por ");
    Serial.print(tempo_restante);
    Serial.println("ms");
  } else {
    Serial.println("[STATUS] Trigger pronto - aceitando comandos");
  }
  
  // Blink de feedback
  digitalWrite(LED_PIN, HIGH); delay(50);
  digitalWrite(LED_PIN, LOW);
}

void blinkBoot() {
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH); delay(150);
    digitalWrite(LED_PIN, LOW);  delay(150);
  }
}

/*
 * Ligacoes:
 * - GPIO19 -> seu atuador/LED esquerdo (com resistor se for LED)
 * - GPIO22 -> seu atuador/LED direito (com resistor se for LED)
 * - GND em comum com o periférico
 *
 * Dicas:
 * - Use Serial Monitor em 115200 baud com "Newline" como line ending
 * - O sistema bloqueia automaticamente por 3 segundos após receber comando
 * - Use PING para verificar status sem ativar bloqueio
 * - O bloqueio impede NOVOS sinais, mas mantém o pino ativado pelo tempo programado
 */
