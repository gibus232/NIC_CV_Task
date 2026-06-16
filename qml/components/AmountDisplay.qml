import QtQuick
import QtQuick.Controls

Item {
    id: root
    property double amount: 0
    property bool isExpense: true
    signal amountChanged(double value)

    height: 100

    property string inputText: "0"

    Column {
        anchors.centerIn: parent
        spacing: 4

        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            text: root.isExpense ? "Расход" : "Доход"
            color: root.isExpense ? "#EF9A9A" : "#A5D6A7"
            font.pixelSize: 14
        }

        Row {
            anchors.horizontalCenter: parent.horizontalCenter
            spacing: 4

            Text {
                anchors.verticalCenter: parent.verticalCenter
                text: root.isExpense ? "−" : "+"
                color: root.isExpense ? "#EF5350" : "#66BB6A"
                font.pixelSize: 32; font.weight: Font.Light
            }

            TextInput {
                id: inp
                anchors.verticalCenter: parent.verticalCenter
                text: root.inputText
                color: root.isExpense ? "#EF5350" : "#66BB6A"
                font.pixelSize: 48; font.weight: Font.Light
                inputMethodHints: Qt.ImhFormattedNumbersOnly
                validator: RegularExpressionValidator { regularExpression: /^\d{0,8}(\.\d{0,2})?$/ }
                onTextChanged: {
                    root.inputText = text
                    root.amount = parseFloat(text) || 0
                    root.amountChanged(root.amount)
                }
                cursorVisible: activeFocus
            }

            Text {
                anchors.verticalCenter: parent.verticalCenter
                text: " ₽"
                color: Qt.rgba(1,1,1,0.55)
                font.pixelSize: 28; font.weight: Font.Light
            }
        }
    }

    function clear() { inp.text = "0"; root.amount = 0 }
    function setValue(v) { inp.text = v > 0 ? v.toString() : "0"; root.amount = v }
}
