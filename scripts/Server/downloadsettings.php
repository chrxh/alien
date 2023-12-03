<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $id = $_GET["id"];

    if ($response = $db->query("SELECT sim.id as id, sim.settings as settings FROM simulation sim where ID=$id")) {
        $obj = $response->fetch_object();
        echo $obj->settings;
    }
    else {
        echo json_encode(["result"=>false]);
    }

    $db->commit();
    $db->close();
?>