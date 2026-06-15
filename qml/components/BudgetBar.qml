import QtQuick

Rectangle {
    id: root
    property string label: ""
    property color barColor: "#1565C0"
    property double spent: 0
    property double budget: 0
    property bool overBudget: budget > 0 && spent > budget
    property double progress: budget > 0 ? Math.min(spent / budget, 1.0) : 0.0

    height: 76
    radius: 16
    color: Qt.rgba(1, 1, 1, 0.07)
    border.color: Qt.rgba(1, 1, 1, 0.15)
    border.width: 1

    Column {
        anchors { fill: parent; margins: 14 }
        spacing: 8

        Row {
            width: parent.width
            Text {
                text: root.label
                color: "white"; font.pixelSize: 14; font.weight: Font.Medium
                width: parent.width * 0.6; elide: Text.ElideRight
            }
            Text {
                text: Number(root.spent).toLocaleString(Qt.locale("ru_RU"),"f",0) +
                      " / " + Number(root.budget).toLocaleString(Qt.locale("ru_RU"),"f",0) + " ₽"
                color: root.overBudget ? "#EF5350" : Qt.rgba(1,1,1,0.65)
                font.pixelSize: 13
                width: parent.width * 0.4
                horizontalAlignment: Text.AlignRight
            }
        }

        Rectangle {
            width: parent.width; height: 8; radius: 4
            color: Qt.rgba(1,1,1,0.12)

            Rectangle {
                width: parent.width * root.progress
                height: parent.height; radius: parent.radius
                color: root.overBudget ? "#EF5350" : root.barColor

                Rectangle {
                    anchors { top: parent.top; left: parent.left; right: parent.right; margins: 0 }
                    height: parent.height / 2; radius: parent.radius
                    color: Qt.rgba(1,1,1,0.35)
                }

                Behavior on width { NumberAnimation { duration: 700; easing.type: Easing.OutCubic } }
            }
        }
    }
}
