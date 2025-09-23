/*
 * ESP32 — TRIGGERS via Serial
 * LEFT  -> GPIO18  (acende LED no 18)
 * RIGHT -> GPIO22
 */

#define TRIGGER_LEFT_PIN   19   // <-- pedido: LEFT no GPIO18
#define TRIGGER_RIGHT_PIN  22   // <-- pedido: RIGHT no GPIO22
#define LED_PIN             2   // LED onboard (a maioria dos ESP32 usa GPIO2)

#define BAUD_RATE          115200
#define TRIGGER_DURATION   100   // ms

void setup() {
  Serial.begin(BAUD_RATE);

  pinMode(TRIGGER_LEFT_PIN, OUTPUT);
  pinMode(TRIGGER_RIGHT_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);

  digitalWrite(TRIGGER_LEFT_PIN, LOW);
  digitalWrite(TRIGGER_RIGHT_PIN, LOW);
  digitalWrite(LED_PIN, LOW);

  Serial.println("ESP32 BCI Trigger System");
  Serial.println("========================");
  Serial.println("Comandos:");
  Serial.println("- TRIGGER_LEFT  | LEFT  | L");
  Serial.println("- TRIGGER_RIGHT | RIGHT | R");
  Serial.println("- PING");
  Serial.println("Use newline (\\n) no final do comando.");
  blinkBoot();
}

void loop() {
  static String buf;

  // lê tudo que chegar e monta uma linha até '\n'
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      buf.trim();                // remove espacos e \r
      if (buf.length() > 0) {
        processCommand(buf);
      }
      buf = "";                  // zera buffer pra próxima linha
    } else {
      buf += c;
    }
  }
}

void processCommand(String command) {
  command.trim();
  command.toUpperCase();

  Serial.print("Comando recebido: ");
  Serial.println(command);

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
    Serial.print("Erro: Comando desconhecido - ");
    Serial.println(command);
  }
}

void executeTriggerLeft() {
  Serial.println("Executando TRIGGER_LEFT (GPIO19)");
  digitalWrite(TRIGGER_LEFT_PIN, HIGH);
  digitalWrite(LED_PIN, HIGH);
  delay(TRIGGER_DURATION);
  digitalWrite(TRIGGER_LEFT_PIN, LOW);
  digitalWrite(LED_PIN, LOW);
  Serial.println("TRIGGER_LEFT finalizado");
}

void executeTriggerRight() {
  Serial.println("Executando TRIGGER_RIGHT (GPIO22)");
  digitalWrite(TRIGGER_RIGHT_PIN, HIGH);
  digitalWrite(LED_PIN, HIGH);
  delay(TRIGGER_DURATION);
  digitalWrite(TRIGGER_RIGHT_PIN, LOW);
  digitalWrite(LED_PIN, LOW);
  Serial.println("TRIGGER_RIGHT finalizado");
}

void executePing() {
  Serial.println("PONG - ESP32 ativo e funcionando");
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
 * - GPIO18 -> LED (através de resistor 220–330Ω) -> GND
 * - GPIO22 -> seu atuador/LED direito (com resistor se for LED)
 * - GND em comum com o periférico
 *
 * Dicas:
 * - No Serial Monitor use 115200 baud e "Newline" como line ending.
 * - Se não tiver LED no GPIO2 no seu modelo, pode ignorar o LED_PIN (só feedback).
 */
