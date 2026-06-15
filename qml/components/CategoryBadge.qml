import QtQuick

Rectangle {
    id: root
    property string label: ""
    property color badgeColor: "#1565C0"
    property bool selected: false
    signal clicked()

    height: 40
    width: Math.max(90, txt.width + 32)
    radius: height / 2
    color: root.selected ? root.badgeColor : Qt.rgba(1,1,1,0.08)
    border.color: root.selected ? Qt.rgba(1,1,1,0.45) : Qt.rgba(1,1,1,0.20)
    border.width: 1
    Behavior on color { ColorAnimation { duration: 150 } }

    Rectangle {
        anchors { top: parent.top; left: parent.left; right: parent.right; margins: 1 }
        height: parent.height * 0.48; radius: parent.radius
        gradient: Gradient {
            GradientStop { position: 0.0; color: root.selected ? Qt.rgba(1,1,1,0.22) : Qt.rgba(1,1,1,0.07) }
            GradientStop { position: 1.0; color: Qt.rgba(1,1,1,0.00) }
        }
    }

    Text {
        id: txt
        anchors.centerIn: parent
        text: root.label
        color: "white"; font.pixelSize: 13
        font.weight: root.selected ? Font.Medium : Font.Normal
    }

    scale: ma.pressed ? 0.93 : 1.0
    Behavior on scale { NumberAnimation { duration: 80 } }
    MouseArea { id: ma; anchors.fill: parent; onClicked: root.clicked() }
}
