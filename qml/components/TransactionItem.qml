import QtQuick

Rectangle {
    id: root
    property var tx: ({})
    signal deleteRequested(int txId)

    height: 72
    radius: 16
    color: Qt.rgba(1, 1, 1, 0.07)
    border.color: Qt.rgba(1, 1, 1, 0.15)
    border.width: 1

    // Top shine
    Rectangle {
        anchors { top: parent.top; left: parent.left; right: parent.right; margins: 1 }
        height: parent.height * 0.45; radius: parent.radius
        gradient: Gradient {
            GradientStop { position: 0.0; color: Qt.rgba(1,1,1,0.07) }
            GradientStop { position: 1.0; color: Qt.rgba(1,1,1,0.00) }
        }
    }

    Row {
        anchors { fill: parent; leftMargin: 14; rightMargin: 14 }
        spacing: 12

        Rectangle {
            width: 44; height: 44; radius: 22
            anchors.verticalCenter: parent.verticalCenter
            color: root.tx.categoryColor || "#1565C0"
            opacity: 0.85

            Text {
                anchors.centerIn: parent
                text: "₽"
                color: "white"; font.pixelSize: 16; font.weight: Font.Bold
            }
        }

        Column {
            anchors.verticalCenter: parent.verticalCenter
            width: parent.width - 44 - 96 - 28
            spacing: 4

            Text {
                text: root.tx.categoryName || ""
                color: "white"; font.pixelSize: 15; font.weight: Font.Medium
                elide: Text.ElideRight; width: parent.width
            }
            Text {
                text: root.tx.note !== "" ? root.tx.note : (root.tx.date || "")
                color: Qt.rgba(1,1,1,0.55); font.pixelSize: 12
                elide: Text.ElideRight; width: parent.width
            }
        }

        Text {
            anchors.verticalCenter: parent.verticalCenter
            text: (root.tx.isExpense ? "−" : "+") +
                  Number(root.tx.amount || 0).toLocaleString(Qt.locale("ru_RU"), "f", 0) + " ₽"
            color: root.tx.isExpense ? "#EF5350" : "#66BB6A"
            font.pixelSize: 16; font.weight: Font.SemiBold
            width: 96; horizontalAlignment: Text.AlignRight
        }
    }

    MouseArea {
        anchors.fill: parent
        onPressAndHold: root.deleteRequested(root.tx.id || 0)
    }
}
