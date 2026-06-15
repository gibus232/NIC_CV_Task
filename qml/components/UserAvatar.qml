import QtQuick

Rectangle {
    id: root
    property string name: ""
    property string avatarColor: "#1565C0"
    property int size: 56

    width: size; height: size
    radius: size / 2
    gradient: Gradient {
        GradientStop { position: 0.0; color: Qt.lighter(root.avatarColor, 1.4) }
        GradientStop { position: 1.0; color: root.avatarColor }
    }
    border.color: Qt.rgba(1, 1, 1, 0.45)
    border.width: 2

    Rectangle {
        anchors { top: parent.top; left: parent.left; right: parent.right; margins: 1 }
        height: parent.height * 0.48
        radius: parent.radius
        gradient: Gradient {
            GradientStop { position: 0.0; color: Qt.rgba(1, 1, 1, 0.30) }
            GradientStop { position: 1.0; color: Qt.rgba(1, 1, 1, 0.00) }
        }
    }

    Text {
        anchors.centerIn: parent
        text: root.name.length > 0 ? root.name[0].toUpperCase() : "?"
        color: "white"
        font.pixelSize: root.size * 0.38
        font.weight: Font.Bold
    }
}
