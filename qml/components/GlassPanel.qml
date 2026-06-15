import QtQuick

Rectangle {
    radius: 20
    color: Qt.rgba(1, 1, 1, 0.09)
    border.color: Qt.rgba(1, 1, 1, 0.25)
    border.width: 1

    Rectangle {
        anchors { top: parent.top; left: parent.left; right: parent.right; margins: 1 }
        height: parent.height * 0.45
        radius: parent.radius
        gradient: Gradient {
            GradientStop { position: 0.0; color: Qt.rgba(1, 1, 1, 0.11) }
            GradientStop { position: 1.0; color: Qt.rgba(1, 1, 1, 0.00) }
        }
    }
}
