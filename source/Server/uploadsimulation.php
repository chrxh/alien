<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $userName = $_POST["userName"];
    $pw = $_POST["password"];

    if (!checkPw($db, $userName, $pw)) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    $obj = $db->query("SELECT u.ID as id FROM user u WHERE u.NAME='".addslashes($userName)."'")->fetch_object();
    if (!$obj) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    $success = false;
    $simName = $_POST['simName'];
    $simDesc = $_POST['simDesc'];
    $width = (int)$_POST['width'];
    $height = (int)$_POST['height'];
    $particles = (int)$_POST['particles'];
    $version = $_POST['version'];
    $content = $_POST['content'];
    $settings = $_POST['settings'];
    $symbolMap = $_POST['symbolMap'];
    if ($db->query("INSERT INTO
                        simulation (ID, USER_ID, NAME, WIDTH, HEIGHT, PARTICLES, VERSION, DESCRIPTION, CONTENT, SETTINGS, SYMBOL_MAP, PICTURE, TIMESTAMP)
                    VALUES
                        (NULL, {$obj->id}, '" . addslashes($simName) . "', $width, $height, $particles, '" . addslashes($version) . "', '"
                        . addslashes($simDesc) . "', '" . addslashes($content) . "', '" . addslashes($settings) . "', '" . addslashes($symbolMap) . "', 'a', NULL)")) {
        $success = true;
    }

    echo json_encode(["result"=>$success]);

    $db->commit();
    $db->close();
?>